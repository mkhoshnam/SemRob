import os
import sys
import json
import time
import base64
import math
import numpy as np
from pathlib import Path
import copy
import re
import random
import cv2

# MuJoCo imports
import mujoco
import mujoco.viewer

import openai
from termcolor import colored

EPS = 1e-6
IGNORE_JOINT_KEYWORDS = {"microjoint"}
EXCLUDE_CABINET_KEYWORDS = {"cab_micro"}
CACHE_PATH = Path.home() / "apartment_plan_cache.json"

# Set your OpenAI API key
os.environ[
    "OPENAI_API_KEY"] = "XXXX"
client = openai.OpenAI()


def _looks_unlimited(lo, hi):
    return abs(hi - lo) < EPS


def _effective_limits(lo, hi, fallback):
    if _looks_unlimited(lo, hi):
        return fallback
    return lo, hi


def _deg2rad_if_needed(x: float) -> float:
    """If the number is outside ±2π it is almost certainly in degrees."""
    return math.radians(x) if abs(x) > 2 * math.pi + 1e-3 else x


class MuJoCoApartmentPlanner:
    def __init__(self, xml_path, cache_path=CACHE_PATH):
        self.xml_path = xml_path
        self.ontology = None
        self.ontology_index = {}
        self.movable_index = {}
        self.digital_index = {}
        self.scene_semantics = {}

        # MuJoCo specific
        self.model = None
        self.data = None
        self.viewer = None
        self.renderer = None

        self.sim_joints = {}
        self.joint_positions = {}
        self.joint_centroid = {}

        # Cache system
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                self.plan_cache = json.load(f)
        else:
            self.plan_cache = {}

    def load_ontology(self, ontology_path):
        """Load ontology from JSON file"""
        ontology_file = Path(ontology_path)
        if not ontology_file.exists():
            return False
        try:
            with open(ontology_file, 'r') as f:
                self.ontology = json.load(f)
            return True
        except Exception as e:
            print(f"Failed to load ontology: {e}")
            return False

    def _iter_ontology_joints(self, data):
        """Iterate through ontology joints"""
        for entry in data.get("joints", []):
            if isinstance(entry, str):
                yield entry, {"sim_name": entry}
            else:
                logical = entry.get("name") or entry.get("sim_name")
                yield logical, entry

        children = (data.get("ontology", {})
                    .get("object", {})
                    .get("children", {}))
        for logical, spec in children.items():
            yield logical, spec

    def build_joint_index(self):
        """Build index of joints from ontology"""
        self.ontology_index.clear()
        if not self.ontology:
            return

        for obj_class, data in self.ontology.items():
            for logical, spec in self._iter_ontology_joints(data):
                sim = spec.get("sim_name") or logical
                if not sim:
                    continue

                self.ontology_index[sim] = {
                    "logical": logical,
                    "obj_class": obj_class,
                    "range": spec.get("range", [-1.57, 1.57]),
                    "unit": spec.get("unit", "rad"),
                    "type": spec.get("type", "revolute"),
                    "affordances": spec.get("affordances", []),
                    "category": (
                        "door" if logical.lower().endswith("hinge")
                        else "drawer" if "slide" in logical.lower()
                        else "other"
                    ),
                }

        for meta in self.ontology_index.values():
            lo, hi = meta["range"]
            if _looks_unlimited(lo, hi):
                meta["range"] = [-1.57, 1.57]

    def setup_environment(self):
        """Setup MuJoCo environment with the apartment XML"""
        try:
            print(f"Loading MuJoCo model from: {self.xml_path}")

            # Load the model
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)

            # Initialize the data
            mujoco.mj_forward(self.model, self.data)

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

            # Discover joints and objects
            self.discover_simulation_joints()
            self.discover_scene_bodies()
            self.augment_ontology_with_scene_joints()

            # Set initial camera position
            self.set_camera_view()

            return True

        except Exception as e:
            print(colored(f"Failed to setup environment: {e}", "red"))
            return False

    def set_camera_view(self):
        """Set a good initial camera view"""
        if self.viewer is None:
            return

        # Set camera to get a good overview of the apartment
        self.viewer.cam.distance = 5.0
        self.viewer.cam.azimuth = 45
        self.viewer.cam.elevation = -20
        self.viewer.cam.lookat[:] = [0, 0, 1.0]

    def discover_simulation_joints(self):
        """Discover all joints in the simulation"""
        self.sim_joints.clear()
        self.joint_positions.clear()
        self.joint_centroid.clear()

        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name and name not in IGNORE_JOINT_KEYWORDS:
                self.sim_joints[name] = i
                self.joint_positions[name] = self.data.qpos[i]

                # Get joint body position for centroid
                body_id = self.model.jnt_bodyid[i]
                self.joint_centroid[name] = self.data.xpos[body_id].copy()

        print(f"Discovered {len(self.sim_joints)} joints in simulation")

    def discover_scene_bodies(self):
        """Discover movable objects in the scene"""
        self.movable_index.clear()

        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if not name or name == "world":
                continue

            # Check if body has a free joint (indicating it's movable)
            body_jntadr = self.model.body_jntadr[i]
            body_jntnum = self.model.body_jntnum[i]

            for j in range(body_jntnum):
                joint_id = body_jntadr + j
                if joint_id < self.model.njnt:
                    joint_type = self.model.jnt_type[joint_id]
                    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                        self.movable_index[name] = {
                            "logical": name,
                            "sim_body": name,
                            "affordances": ["pickable", "placeable"],
                        }
                        break

        print(f"Discovered {len(self.movable_index)} movable objects")

    def augment_ontology_with_scene_joints(self):
        """Add scene joints to ontology if not already present"""
        added = 0

        for name in self.sim_joints:
            if name in self.ontology_index:
                continue

            # Classify joint by name patterns
            if re.search(r"door|hinge", name, re.IGNORECASE):
                joint_type = "door"
                affordances = ["openable"]
            elif re.search(r"drawer|slide", name, re.IGNORECASE):
                joint_type = "drawer"
                affordances = ["openable"]
            elif re.search(r"knob|handle", name, re.IGNORECASE):
                joint_type = "knob"
                affordances = ["rotatable"]
            else:
                joint_type = "other"
                affordances = ["movable"]

            # Get joint limits from MuJoCo
            joint_id = self.sim_joints[name]
            lo, hi = self.model.jnt_range[joint_id]

            if _looks_unlimited(lo, hi):
                lo, hi = [-1.57, 1.57]  # Default range

            spec = {
                "sim_name": name,
                "type": "revolute",
                "range": [float(lo), float(hi)],
                "unit": "rad",
                "affordances": affordances,
                "preferred_open": hi if abs(hi) > abs(lo) else lo,
                "category": joint_type
            }

            # Add to ontology
            if joint_type not in self.ontology:
                self.ontology[joint_type] = {
                    "ontology": {"object": {"children": {}}}
                }

            self.ontology[joint_type]["ontology"]["object"]["children"][name] = spec
            added += 1

        if added > 0:
            print(f"Augmented ontology with {added} scene joints")
            self.build_joint_index()

    def list_cameras(self):
        """List available cameras"""
        cameras = []
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                cameras.append(name)

        print(f"Available cameras: {cameras}")
        return cameras

    def take_screenshot(self, width=640, height=480, save=True):
        """Take a screenshot of the current view"""
        try:
            # Update the renderer
            self.renderer.update_scene(self.data)

            # Render the scene
            pixels = self.renderer.render()

            # Convert to RGB (MuJoCo returns RGB)
            img = pixels

            if save:
                out_path = Path.home() / "Pictures" / "results" / "apartment_view.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert to BGR for OpenCV
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), bgr)
                print(colored(f"Screenshot saved to {out_path}", "cyan"))

            return img

        except Exception as e:
            print(colored(f"Failed to take screenshot: {e}", "red"))
            return None

    def move_joint_to_position(self, joint_name, target_position):
        """Move a joint to target position with smooth interpolation"""
        try:
            if joint_name not in self.sim_joints:
                print(colored(f"Joint '{joint_name}' not found", "red"))
                return False

            joint_id = self.sim_joints[joint_name]
            current_pos = self.data.qpos[joint_id]

            # Smooth interpolation
            steps = 30
            for step in range(steps):
                alpha = (step + 1) / steps
                new_pos = current_pos + alpha * (target_position - current_pos)

                self.data.qpos[joint_id] = new_pos
                mujoco.mj_forward(self.model, self.data)

                # Update viewer
                if self.viewer is not None:
                    self.viewer.sync()

                time.sleep(0.03)

            # Update position tracking
            self.joint_positions[joint_name] = self.data.qpos[joint_id]
            print(colored(f" Moved {joint_name} to {target_position:.3f}", "green"))
            return True

        except Exception as e:
            print(colored(f"Failed to move joint '{joint_name}': {e}", "red"))
            return False

    def place_object(self, object_name, pose):
        """Place an object at specified pose"""
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
            if body_id == -1:
                print(colored(f"Object '{object_name}' not found", "red"))
                return False

            pos, quat = np.array(pose[:3]), np.array(pose[3:7])

            # Find the free joint for this body
            body_jntadr = self.model.body_jntadr[body_id]

            # Set position and orientation
            self.data.qpos[body_jntadr:body_jntadr + 3] = pos
            self.data.qpos[body_jntadr + 3:body_jntadr + 7] = quat

            mujoco.mj_forward(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

            print(colored(f" Placed {object_name} at {pos}", "green"))
            return True

        except Exception as e:
            print(colored(f"Failed to place object '{object_name}': {e}", "red"))
            return False

    def ask_llm_for_plan(self, command):
        """Get plan from LLM based on command"""
        if command in self.plan_cache:
            return self.plan_cache[command]

        # Filter out bad joints
        def _is_bad(j):
            return any(keyword in j.lower() for keyword in IGNORE_JOINT_KEYWORDS)

        all_joints = {
            j: {
                **self.ontology_index.get(j, {
                    "range": [-1.57, 1.57],
                    "type": "revolute",
                    "affordances": ["movable"],
                    "category": "other"
                }),
                "position": [float(x) for x in self.joint_centroid[j]]
            }
            for j in self.sim_joints if not _is_bad(j)
        }

        # Create LLM prompt
        system_prompt = f"""
You are an expert robot task planner for an apartment environment.

Available JOINTS (continuous control):
{json.dumps(all_joints, indent=2)}

Available OBJECTS (movable items):
{json.dumps(self.movable_index, indent=2)}

SCENE_HINTS:
{json.dumps(self.scene_semantics, indent=2)}

Return ONLY valid JSON in this exact structure:
{{
  "steps": [
    {{"action": "set_joint", "joint": "joint_name", "target": value_in_radians}},
    {{"action": "place", "object": "object_name", "pose": [x,y,z, qw,qx,qy,qz]}}
  ],
  "success_criteria": "description of what success looks like"
}}

Guidelines:
- All joint angles must be in radians
- Use joint ranges to determine appropriate target values
- For "open" commands, typically use the larger absolute value from the range
- For "close" commands, typically use the smaller value (often 0)
- Be specific about which joints to move based on the command
- Consider joint names and categories to match user intent
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Command: {command}"}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]

            plan = json.loads(content)

            # Validate plan structure
            if not isinstance(plan.get("steps"), list):
                print(colored("Invalid plan format - steps must be a list", "red"))
                return None

            # Cache the plan
            self.plan_cache[command] = plan
            with open(self.cache_path, 'w') as f:
                json.dump(self.plan_cache, f, indent=2)

            return plan

        except Exception as e:
            print(colored(f"Failed to get plan from LLM: {e}", "red"))
            return None

    def execute_llm_plan(self, plan):
        """Execute the plan generated by LLM"""
        if not plan or "steps" not in plan:
            print(colored("Plan missing or has no 'steps' field", "red"))
            return False

        for i, step in enumerate(plan["steps"]):
            action = step.get("action", "set_joint")
            print(f"Executing step {i + 1}: {action}")

            if action == "set_joint":
                joint_name = step["joint"]
                target = step["target"]

                # Convert from degrees if needed
                target = _deg2rad_if_needed(target)

                # Clamp to joint limits
                if joint_name in self.sim_joints:
                    joint_id = self.sim_joints[joint_name]
                    lo, hi = self.model.jnt_range[joint_id]
                    target = np.clip(target, lo, hi)

                if not self.move_joint_to_position(joint_name, target):
                    return False

            elif action == "place":
                object_name = step["object"]
                pose = step["pose"]
                if not self.place_object(object_name, pose):
                    return False

            else:
                print(colored(f"Unknown action: {action}", "red"))
                return False

        return True

    def validate_with_vlm(self, command, plan):
        """Validate task completion with Vision Language Model"""
        print(colored("Capturing screenshot for VLM validation...", "cyan"))
        time.sleep(2)  # Let scene settle

        screenshot = self.take_screenshot()
        if screenshot is None:
            print(colored("Could not capture screenshot for VLM", "red"))
            return

        # Encode for API
        success, buf = cv2.imencode('.png', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if not success:
            return

        question = (
            f"Has this task been completed?\n"
            f"• Command: \"{command}\"\n"
            f"• Success criteria: {plan.get('success_criteria', 'Task completion')}\n"
            "Please answer Yes or No and briefly explain what you see."
        )

        b64png = "data:image/png;base64," + base64.b64encode(buf).decode()

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": b64png}}
                    ]
                }]
            )

            answer = response.choices[0].message.content.strip()
            success = answer.lower().startswith("yes")
            color = "green" if success else "red"
            print(colored(f"VLM validation: {answer}", color))

        except Exception as e:
            print(colored(f"VLM validation failed: {e}", "red"))

    def validate_task(self, plan, command):
        """Validate task execution"""
        if "steps" not in plan:
            return

        errors = []

        for step in plan["steps"]:
            action = step.get("action", "set_joint")

            if action == "set_joint":
                joint_name = step["joint"]
                target = step["target"]
                target = _deg2rad_if_needed(target)

                if joint_name in self.sim_joints:
                    joint_id = self.sim_joints[joint_name]
                    actual = self.data.qpos[joint_id]
                    if abs(actual - target) > 0.05:  # 5cm tolerance
                        errors.append(("joint", joint_name, target, float(actual)))

        if not errors:
            print(colored(" Task validation passed", "green"))
            self.validate_with_vlm(command, plan)
        else:
            print(colored(f"Task validation failed: {errors}", "yellow"))
            self.validate_with_vlm(command, plan)

    def run_interactive_session(self):
        """Run the interactive LLM-driven session"""
        print(colored(" MuJoCo Apartment LLM Task Planner", "cyan"))
        print("=" * 50)

        # Try to load ontology
        possible_ontology_paths = [
            "ontology.json",
            "apartment_ontology.json",
            Path.home() / "apartment_ontology.json"
        ]

        ontology_loaded = False
        for path in possible_ontology_paths:
            if self.load_ontology(path):
                print(f" Loaded ontology from: {path}")
                ontology_loaded = True
                break

        if not ontology_loaded:
            print(colored("⚠ No ontology file found, using scene discovery only", "yellow"))
            self.ontology = {}

        self.build_joint_index()

        if not self.setup_environment():
            print(colored("✗ Failed to setup environment", "red"))
            return

        print(f"\n📊 Environment Summary:")
        print(f"  • Joints: {len(self.sim_joints)}")
        print(f"  • Movable objects: {len(self.movable_index)}")
        print(f"  • Ontology entries: {len(self.ontology_index)}")

        if self.sim_joints:
            print(f"\n🔧 Available joints:")
            for joint_name in sorted(self.sim_joints.keys())[:10]:  # Show first 10
                print(f"  • {joint_name}")
            if len(self.sim_joints) > 10:
                print(f" and {len(self.sim_joints) - 10} more")

        while True:
            try:
                command = input(colored("\n Your command: ", "yellow")).strip()

                if command.lower() in ['quit', 'exit', 'q']:
                    break

                if not command:
                    continue

                print(colored(f"\n Planning: '{command}'", "blue"))

                plan = self.ask_llm_for_plan(command)
                if plan:
                    print(colored(f" Plan: {json.dumps(plan, indent=2)}", "blue"))

                    if self.execute_llm_plan(plan):
                        self.validate_task(plan, command)
                    else:
                        # Remove failed plan from cache
                        self.plan_cache.pop(command, None)
                        with open(self.cache_path, "w") as f:
                            json.dump(self.plan_cache, f, indent=2)
                        print(colored("✗ Plan execution failed", "red"))
                else:
                    print(colored("✗ Could not create a plan", "red"))

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(colored(f"Error: {e}", "red"))

        # Cleaning
        if self.viewer:
            self.viewer.close()
        print(colored(" Session ended", "green"))


def main():
    apartment_xml = "/media/mohammad/5f7c6d23-fd63-41c6-a822-d02e7c729060/test/apartment_full/mjcf/ApartmentReduce.xml"

    if not Path(apartment_xml).exists():
        print(colored(f"❌ XML file not found: {apartment_xml}", "red"))
        print("Please update the path in main() function")
        return

    # Create and run planner
    planner = MuJoCoApartmentPlanner(apartment_xml)
    planner.run_interactive_session()

if __name__ == "__main__":
    main()
