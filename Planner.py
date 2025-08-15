import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
import sys
import json
import time
import base64
import math  # NEW: Added for degree/radian conversion
import numpy as np
from pathlib import Path
import copy
import re
import random
import cv2

import os
import sys
import json
import time
import base64
import numpy as np
from pathlib import Path
import copy
import re
import random
import cv2

CACHE_PATH = Path.home() / "robocasa_plan_cache.json"


# Try to find RoboCasa installation automatically
def find_robocasa_root():
    """Find RoboCasa installation directory"""
    possible_paths = [
        "/media/mohammad/5f7c6d23-fd63-41c6-a822-d02e7c729060/robocasa",
        os.path.expanduser("~/robocasa"),
        os.path.expanduser("~/RoboCasa"),
        os.path.expanduser("~/PhD/robocasa"),
        "/opt/robocasa",
        "/usr/local/robocasa"
    ]

    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "robocasa")):
            return path

    # If not found, try to detect from installed packages
    try:
        import robocasa
        return os.path.dirname(os.path.dirname(robocasa.__file__))
    except ImportError:
        pass

    return None


ROBOCASA_ROOT = find_robocasa_root()
if ROBOCASA_ROOT is None:
    print("❌ Could not find RoboCasa installation!")
    print("Please check that RoboCasa is installed and update ROBOCASA_ROOT path.")
    print("Try running: pip install -e /path/to/robocasa")
    sys.exit(1)

print(f" Found RoboCasa at: {ROBOCASA_ROOT}")
sys.path.insert(0, ROBOCASA_ROOT)
os.chdir(ROBOCASA_ROOT)

import robosuite
from robosuite import load_controller_config
from termcolor import colored
import robocasa
from robocasa.models.scenes.scene_registry import LayoutType, StyleType
from robocasa.models.fixtures import FixtureType
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.objects.objects import MJCFObject

import openai

EPS = 1e-6

IGNORE_JOINT_KEYWORDS = {"microjoint"}

EXCLUDE_CABINET_KEYWORDS = {"cab_micro"}


def _looks_unlimited(lo, hi):
    return abs(hi - lo) < EPS


def _effective_limits(lo, hi, fallback):
    if _looks_unlimited(lo, hi):
        return fallback
    return lo, hi

def _deg2rad_if_needed(x: float) -> float:
    """If the number is outside ±2π it is almost certainly in degrees."""
    return math.radians(x) if abs(x) > 2 * math.pi + 1e-3 else x

os.environ[
    "OPENAI_API_KEY"] = "XXXX"
client = openai.OpenAI()


class LLMTaskPlanner:
    def __init__(self, cache_path=CACHE_PATH):
        self.ontology = None
        self.ontology_index = {}
        self.movable_index = {}
        self.digital_index = {}
        self.scene_semantics = {}
        self.env = None
        self.sim_joints = {}
        self.joint_positions = {}
        self._mw_light_gid = None
        self._render_cam = None

        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                self.plan_cache = json.load(f)
        else:
            self.plan_cache = {}

    def _cache_microwave_light(self):
        if self._mw_light_gid is not None:
            return

        for gid in range(self.env.sim.model.ngeom):
            gname = self.env.sim.model.geom_id2name(gid)
            if gname and "microwave_interior_light" in gname:
                self._mw_light_gid = gid
                return

    def _set_microwave_light(self, on: bool):
        self._cache_microwave_light()
        if self._mw_light_gid is None:
            return
        rgba = np.array([1, 0.95, 0.7, 1.0 if on else 0.0])
        self.env.sim.model.geom_rgba[self._mw_light_gid] = rgba

    def _live_cam(self):
        vw = getattr(self.env, "viewer", None)
        if vw is None:
            return None
        if hasattr(vw, "viewer") and hasattr(vw.viewer, "cam"):  # robosuite
            return vw.viewer.cam
        if hasattr(vw, "cam"):  # mujoco-py
            return vw.cam
        winctx = getattr(vw, "_render_context_window", None)  # MuJoCo ≥2.3
        if winctx is not None and hasattr(winctx, "cam"):
            return winctx.cam
        return None


    def _init_working_camera(self, focus_body="microwave_left_group_main"):
        vw = self.env.viewer
        cam = (getattr(vw, "_render_context_window", None) and
               getattr(vw._render_context_window, "cam", None)) or \
              getattr(vw, "cam", None)
        if cam is None:
            return

        try:
            body_id = self.env.sim.model.body_name2id(focus_body)
            cam.lookat[:] = self.env.sim.data.body_xpos[body_id]
        except (KeyError, ValueError):
            cam.lookat[:] = np.array([0., 0., 0.8])


        cam.distance = 2.2
        cam.elevation = -18
        cam.azimuth = 90
        cam.fovy = 45

    def load_ontology(self, ontology_path):
        ontology_file = Path(ontology_path)

        if not ontology_file.exists():
            return False

        try:
            with open(ontology_file, 'r') as f:
                self.ontology = json.load(f)
            return True
        except Exception as e:
            return False

    def _iter_ontology_joints(self, data):

        for entry in data.get("joints", []):
            if isinstance(entry, str):
                yield entry, {"sim_name": entry}
            else:  # dict
                logical = entry.get("name") or entry.get("sim_name")
                yield logical, entry

        children = (data.get("ontology", {})
                    .get("object", {})
                    .get("children", {}))
        for logical, spec in children.items():
            yield logical, spec

    def build_joint_index(self) -> None:
        """
        Fill self.ontology_index  so that  sim_joint_name  >  metadata.

        * Any entry without a name is ignored.
        * Range defaults to ±90 ° (±1.57 rad) if neither MuJoCo nor ontology gives one.
        """
        self.ontology_index.clear()
        if not self.ontology:
            return

        for obj_class, data in self.ontology.items():
            for logical, spec in self._iter_ontology_joints(data):
                sim = spec.get("sim_name") or logical  # ← fallback!
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

    def _get_template_by_keyword(self, keyword):
        keyword = keyword.lower()
        for obj, data in self.ontology.items():
            for child in data["ontology"]["object"]["children"].values():
                sim_name = child.get("sim_name", "")
                if keyword in sim_name.lower():
                    return copy.deepcopy(child), obj
        return None, None

    def _get_template_for_scene_name(self, scene_name):
        """
        If any ontology child's sim_name is a suffix of scene_name, return it.
        """
        scene_l = scene_name.lower()
        for obj_class, data in self.ontology.items():
            for child in data["ontology"]["object"]["children"].values():
                sim = child.get("sim_name", "")
                if sim and scene_l.endswith(sim.lower()):
                    return copy.deepcopy(child), obj_class
        return None, None

    def augment_ontology_with_scene_joints(self, scene_joint_names):
        added = 0
        model = self.env.sim.model
        for name in scene_joint_names:
            if name in self.ontology_index:
                continue

            template, obj_class = self._get_template_for_scene_name(name)
            if template:
                try:
                    joint_id = model.joint_name2id(name)
                    lo, hi = model.jnt_range[joint_id]
                except:
                    lo, hi = template.get("range", [-1.0, 1.0])

                spec = copy.deepcopy(template)
                spec["sim_name"] = name
                spec["range"] = [float(lo), float(hi)]
                spec["preferred_open"] = hi if abs(hi) > abs(lo) else lo

                if obj_class not in self.ontology:
                    self.ontology[obj_class] = {
                        "ontology": {"object": {"children": {}}}
                    }
                self.ontology[obj_class]["ontology"]["object"]["children"][name] = spec
                added += 1
                continue

            if re.search(r"doorhinge|leftdoor|rightdoor|microjoint", name):
                kw = "door"
            elif re.search(r"slidejoint|drawer", name):
                kw = "drawer"
            elif re.search(r"knob", name):
                kw = "knob"
            elif re.search(r"spout", name):
                kw = "spout"  # new fallback
            elif re.search(r"handle", name):
                kw = "handle"  # new fallback
            else:
                continue

            template, obj_class = self._get_template_by_keyword(kw)
            if template is None:
                # create a *minimal* generic spec so we stay ontology-only
                template = {
                    "sim_name": kw,
                    "type": "revolute",
                    "range": [-1.57, 1.57],
                    "unit": "rad",
                    "affordances": ["rotatable" if kw == "knob" else "openable"],
                }
                obj_class = "generated"

            try:
                joint_id = model.joint_name2id(name)
                lo_sim, hi_sim = model.jnt_range[joint_id]
                # if MuJoCo gives [0,0], fall back to the template's range
                if _looks_unlimited(lo_sim, hi_sim):
                    lo, hi = template.get("range", [-1.57, 1.57])
                else:
                    lo, hi = lo_sim, hi_sim
            except:
                lo, hi = template.get("range", [-1.57, 1.57])

            spec = copy.deepcopy(template)
            spec["sim_name"] = name
            spec["range"] = [float(lo), float(hi)]

            spec["preferred_open"] = hi if abs(hi) > abs(lo) else lo

            if obj_class not in self.ontology:
                # create a synthetic object bucket once
                self.ontology[obj_class] = {
                    "ontology": {"object": {"children": {}}}
                }
            self.ontology[obj_class]["ontology"]["object"]["children"][name] = spec
            added += 1

        if added:
            # print(colored(f" Augmented ontology with {added} scene-specific joints", "cyan"))
            pass

        self.build_joint_index()

    def discover_special_fixtures(self):
        self.digital_index = {}
        try:
            mw = self.env.get_fixture(FixtureType.MICROWAVE)
        except Exception:
            mw = None

        if mw is not None:
            for btn, logic in (("start_button", "turn_on"), ("stop_button", "turn_off")):
                sim_name = f"microwave_{btn}"
                self.digital_index[sim_name] = {
                    "logical": sim_name,
                    "obj_class": "microwave_button",
                    "fixture": "microwave",
                    "fixture_button": btn,
                    "affordances": ["pressable", logic],
                    "range": [0, 1],
                    "unit": "binary",
                    "type": "button",
                }
                self.ontology_index[sim_name] = self.digital_index[sim_name]

        try:
            cm = self.env.get_fixture("coffee_machine")
        except Exception:
            cm = None

        if cm is not None:
            for btn, logic in (("start_button", "brew"),):
                sim_name = f"coffee_machine_{btn}"
                self.digital_index[sim_name] = {
                    "logical": sim_name,
                    "obj_class": "coffee_button",
                    "fixture": "coffee_machine",
                    "fixture_button": btn,
                    "affordances": ["pressable", logic],
                    "range": [0, 1],
                    "unit": "binary",
                    "type": "button",
                }
                self.ontology_index[sim_name] = self.digital_index[sim_name]

    def verify_joint_ranges(self):
        """Verify that stovetop knobs have proper ranges for debugging"""
        for joint_name in self.sim_joints:
            if "knob" in joint_name.lower():
                joint_range = self.ontology_index[joint_name]["range"]
                # print(f" {joint_name}: range = {joint_range}")
                if _looks_unlimited(*joint_range):
                    print(f"   Still has degenerate range!")
                else:
                    # print(f"   Has proper range for turning")
                    pass

    def list_cameras(self):
        cams = [self.env.sim.model.camera_id2name(i)
                for i in range(self.env.sim.model.ncam)]
        print("\n Available cameras:")
        for c in cams:
            print(f"  • {c}")
        return cams

    def spawn_kitchen_object(self, object_type, position, name=None):
        try:
            mjcf_kwargs, info = sample_kitchen_object(groups=object_type)

            if name is None:
                name = f"{object_type}_{random.randint(1000, 9999)}"

            food_obj = MJCFObject(name=name, **mjcf_kwargs)

            try:
                table_z = self.env.get_fixture("island_counter").top_z
                spawn_pos = [position[0], position[1], table_z + 0.02]
            except:
                spawn_pos = list(position)
            spawn_quat = [1, 0, 0, 0]

            self.env.add_object(food_obj, pos=spawn_pos, quat=spawn_quat)

            # Initialize qpos for free joint
            root_body_id = self.env.sim.model.body_name2id(food_obj.root_body)
            adr = self.env.sim.model.body_dofadr[root_body_id]
            self.env.sim.data.qpos[adr: adr + 7] = spawn_pos + spawn_quat
            self.env.sim.forward()
            self.env.render()
            time.sleep(2.0)

            # Add to movable index
            self.movable_index[food_obj.root_body] = {
                "logical": food_obj.root_body,
                "sim_body": food_obj.root_body,
                "affordances": ["pickable", "placeable"],
            }

            print(f" spawned {food_obj.root_body}")
            print("spawned:", food_obj.root_body)
            print("movable index now:", list(self.movable_index))

            for n in self.env.sim.model.body_names:
                if n.startswith(("apple", "bread", "steak")):
                    print("play", n)

            return food_obj

        except Exception as e:
            print(colored(f"Failed to spawn {object_type}: {e}", "red"))
            return None

    def _pick_camera(self, preferred=("front", "agentview", "overhead")):
        """Return a camera name that exists in the current model."""
        cams = [self.env.sim.model.camera_id2name(i)
                for i in range(self.env.sim.model.ncam)]
        # 1) anything that contains one of the preferred keywords
        for key in preferred:
            for c in cams:
                if key in c.lower():
                    return c
        return cams[0]

    def _pick_camera(self, preferred=("front", "agentview", "overhead")):
        cams = [self.env.sim.model.camera_id2name(i) for i in range(self.env.sim.model.ncam)]
        for key in preferred:
            for c in cams:
                if key in c.lower():
                    return c
        return cams[0]

    @staticmethod
    def _copy_cam(dst, src):
        if src is None or dst is None:
            return
        for k in ("lookat", "distance", "azimuth", "elevation", "fovy", "type", "trackbodyid", "fixedcamid"):
            if hasattr(src, k) and hasattr(dst, k):
                val = getattr(src, k)
                if isinstance(val, np.ndarray):
                    getattr(dst, k)[:] = val
                else:
                    setattr(dst, k, val)

    def take_screenshot(self, w=640, h=480, save=True):

        import cv2
        import numpy as np
        from pathlib import Path

        if self.env.sim._render_context_offscreen is None:
            try:
                from mujoco_py import MjRenderContextOffscreen
                self.env.sim._render_context_offscreen = MjRenderContextOffscreen(self.env.sim, 0)
                using_pymy = True
            except ImportError:
                import mujoco
                self.env.sim._render_context_offscreen = mujoco.MjRenderContextOffscreen(self.env.sim)
                using_pymy = False
        else:
            using_pymy = hasattr(self.env.sim._render_context_offscreen, 'read_pixels')

        off = self.env.sim._render_context_offscreen

        live_cam = self._live_cam()
        if live_cam is None:
            print(colored(" Could not access live viewer camera; falling back to fixed 'frontview'", "yellow"))
            cam_name = self._pick_camera(preferred=("frontview", "agentview"))
            cam_id = self.env.sim.model.camera_name2id(cam_name)
        else:
            cam_id = -1

        if using_pymy:
            if live_cam is not None:
                self._copy_cam(off.cam, live_cam)
        else:
            if live_cam is not None:
                self._copy_cam(off.con.cam, live_cam)
        self.env.render()

        if using_pymy:
            off.render(w, h)
            data = off.read_pixels(h, w, depth=False)
            if isinstance(data, tuple):
                data = data[0]
            img = data[::-1, :, ::-1].copy()
        else:
            img = off.render(width=w, height=h, camera_id=cam_id)

        if img is None:
            raise RuntimeError("Offscreen render failed (returned None)")

        if save:
            out = Path.home() / "Pictures" / "results" / "mujoco_view.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out), bgr)
            print(colored(f" Saved screenshot to {out}", "cyan"))
        return img

    def validate_with_vlm(self, command, plan):
        print(colored("Capturing current viewer screenshot for VLM validation...", "cyan"))
        time.sleep(10)
        screenshot = self.take_screenshot()
        if screenshot is None:
            print(colored(" Could not grab screenshot for VLM", "red"))
            return

        success, buf = cv2.imencode('.png', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        if not success:
            return

        question = (
            f"Has this task been completed?  \n"
            f"• Command: \"{command}\"  \n"
            f"• Success criteria: {plan.get('success_criteria', '<none>')}.  \n"
            "Please answer Yes or No and briefly explain what you see in the image."
        )

        # Encode image as base64 for vision API
        b64png = "data:image/png;base64," + base64.b64encode(buf).decode()

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": b64png}}
                ]
            }]
        )
        answer = resp.choices[0].message.content.strip()
        ok = answer.lower().startswith("yes")
        print(colored(f" VLM validation: {answer}", "green" if ok else "red"))

    def setup_environment(self, layout=None, style=None):
        if layout is None:
            layout = random.choice(range(1, 9))
            # layout = 1
        elif isinstance(layout, LayoutType):
            layout = layout.value

        if style is None:
            style = random.choice(range(0, 10))
            # style = 1
        elif isinstance(style, StyleType):
            style = style.value

        # print(f" Attempting to load layout {layout}, style {style}")

        config = {
            "env_name": "PnPCounterToCab",
            "robots": "PandaMobile",
            "controller_configs": load_controller_config(default_controller="OSC_POSE"),
            "translucent_robot": False,
        }

        try:
            self.env = robosuite.make(
                **config,
                has_renderer=True,
                has_offscreen_renderer=True,
                render_camera=None,
                ignore_done=True,
                use_camera_obs=False,
                control_freq=20,
                renderer="mjviewer",
            )

            obs = self.env.reset()
            self._init_working_camera()
            self._set_microwave_light(False)


            self.list_cameras()
            cams = self.list_cameras()
            self._render_cam = next((c for c in cams if "frontview" in c.lower()), cams[0])

            self.env.render()

            scene_joints = [self.env.sim.model.joint_id2name(i)
                            for i in range(self.env.sim.model.njnt)]

            self.augment_ontology_with_scene_joints(scene_joints)

            self.discover_simulation_joints()
            self.discover_scene_bodies()  # NEW
            self.discover_special_fixtures()  # NEW: discover microwave buttons

            if self.digital_index:
                # print("\nAvailable digital controls in the scene:")
                for ctrl_name in sorted(self.digital_index.keys()):
                    # print(f"  - {ctrl_name}")
                    pass

            # Verify that knob ranges are fixed
            # print("\n Verifying stovetop knob ranges:")
            self.verify_joint_ranges()

            # print(f" Successfully loaded kitchen scene!")
            return True

        except Exception as e:
            print(colored(f" [setup_environment] Exception: {e}", "red"))
            print(f"   Layout: {layout}, Style: {style}")
            if "No such file or directory" in str(e):
                print("   This looks like a file path issue. Checking RoboCasa installation...")
                self._check_robocasa_installation()
            if "'list' object has no attribute 'items'" in str(e):
                print("   Trying different layout combinations...")
                return self.try_different_layouts()
            else:
                return False

    def _check_robocasa_installation(self):
        expected_files = [
            "robocasa/models/assets/scenes/kitchen_styles",
            "robocasa/environments",
            "robocasa/models/scenes"
        ]

        print("\n Checking RoboCasa installation:")
        for file_path in expected_files:
            full_path = os.path.join(ROBOCASA_ROOT, file_path)
            exists = os.path.exists(full_path)
            status = "pass" if exists else "fail"
            print(f"   {status} {file_path}")

        # print(f"\n Suggestions:")
        # print(f"   - Verify RoboCasa is properly installed: pip install -e /path/to/robocasa")
        # print(f"   - Check that all asset files are downloaded")
        # print(f"   - Try running: cd {ROBOCASA_ROOT} && python -m robocasa.scripts.download_kitchen_assets")

    def try_different_layouts(self):
        # Try different valid layout combinations
        working_combinations = [
            (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1),
            (4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1),
            (7, 0), (7, 1), (8, 0), (8, 1)
        ]

        for layout, style in working_combinations:
            try:
                config = {
                    "env_name": "PnPCounterToCab",
                    "robots": "PandaMobile",
                    "controller_configs": load_controller_config(default_controller="OSC_POSE"),
                    "translucent_robot": False,
                }

                self.env = robosuite.make(
                    **config,
                    has_renderer=True,
                    has_offscreen_renderer=True,
                    render_camera=None,
                    ignore_done=True,
                    use_camera_obs=False,
                    control_freq=20,
                    renderer="mjviewer",
                )

                obs = self.env.reset()
                self._set_microwave_light(False)

                self.list_cameras()
                self._render_cam = None

                self.env.render()

                scene_joints = [self.env.sim.model.joint_id2name(i)
                                for i in range(self.env.sim.model.njnt)]

                self.augment_ontology_with_scene_joints(scene_joints)

                self.discover_simulation_joints()
                self.discover_scene_bodies()
                self.discover_special_fixtures()

                if self.digital_index:
                    print("\nAvailable digital controls in the scene:")
                    for ctrl_name in sorted(self.digital_index.keys()):
                        print(f"  - {ctrl_name}")

                # Verify that knob ranges are fixed
                # print("\n Verifying stovetop knob ranges:")
                self.verify_joint_ranges()

                print(colored(f" Successfully loaded layout {layout}, style {style}", "green"))
                return True

            except Exception as e:
                print(colored(f" [try_different_layouts] Layout {layout}, Style {style} failed: {e}", "yellow"))
                if "'list' object has no attribute" in str(e):
                    continue
                else:
                    continue

        print(colored(" All layout combinations failed", "red"))
        return False

    def discover_scene_bodies(self):
        """
        Find individual movable objects by examining geometries instead of just bodies.
        Filters out large objects (furniture) by size.
        """
        self.movable_index = {}
        model = self.env.sim.model
        world = model.body_name2id('world')
        robot_prefix = 'robot0'

        for gi in range(model.ngeom):
            bi = model.geom_bodyid[gi]
            name = model.body_id2name(bi)
            if not name or name.startswith(robot_prefix):
                continue
            if model.body_parentid[bi] != world:
                continue

            size = model.geom_size[gi]
            if np.linalg.norm(size) > 0.25:
                continue

            self.movable_index[name] = {
                "logical": name,
                "sim_body": name,
                "affordances": ["pickable", "placeable"],
            }

        try:
            sink_basin = model.body_name2id('sink_left_group_sink_basin')
            center = self.env.sim.data.body_xpos[sink_basin]
            self.scene_semantics = {
                "sink_center": center.tolist(),
            }
        except:
            self.scene_semantics = {
                "sink_center": [0.3, -0.8, 0.9],
            }

    def discover_simulation_joints(self):
        model = self.env.sim.model
        self.joint_centroid = {}
        for i in range(model.njnt):
            name = model.joint_id2name(i)
            if name and name in self.ontology_index:
                self.sim_joints[name] = i
                self.joint_positions[name] = self.env.sim.data.qpos[i]
                # centroid of the *body* the joint sits on
                b = model.jnt_bodyid[i]
                self.joint_centroid[name] = self.env.sim.data.body_xpos[b].copy()

        # print(" Joints seen in Mujoco but ✗ missing from ontology:")
        missing = [model.joint_id2name(i)
                   for i in range(model.njnt)
                   if model.joint_id2name(i) and model.joint_id2name(i) not in self.ontology_index]
        if missing:
            # print(f"Missing joints: {missing}")
            pass
        else:
            print("All joints are covered by ontology!")

    def _root_free_joint_of(self, body_name):
        m = self.env.sim.model
        b = m.body_name2id(body_name)
        while m.body_parentid[b] != 0:
            b = m.body_parentid[b]
        return m.body_dofadr[b]

    def _teleport_body(self, body_id: int, pos: np.ndarray, quat: np.ndarray = None):
        adr = self._root_free_joint_of(self.env.sim.model.body_id2name(body_id))
        self.env.sim.data.qpos[adr:adr + 3] = pos
        if quat is not None:
            self.env.sim.data.qpos[adr + 3:adr + 7] = quat

    def press_button(self, btn_name: str) -> bool:
        meta = self.digital_index.get(btn_name, {})
        if not meta:
            print(colored(f" Button '{btn_name}' not found in simulation", "red"))
            return False

        try:
            fixture_name = meta["fixture"]
            fx = (self.env.get_fixture(FixtureType.MICROWAVE)
                  if fixture_name == "microwave"
                  else self.env.get_fixture("coffee_machine"))

            print("Turned-on before :", fx.get_state().get("turned_on", False))

            if meta["fixture_button"] == "start_button":
                fx._turned_on = True
            elif meta["fixture_button"] == "stop_button":
                fx._turned_on = False

            self._set_microwave_light(fx.get_state()["turned_on"])

            print("Turned-on after  :", fx.get_state().get("turned_on", False))

            self.env.step(np.zeros(self.env.action_dim))
            self.env.render()

            print(colored(f" Pressed {btn_name}", "green"))
            return True
        except Exception as e:
            print(colored(f" Failed to press {btn_name}: {e}", "red"))
            return False

    def ask_llm_for_plan(self, command):
        if command in self.plan_cache:
            return self.plan_cache[command]

        def _is_bad(j):
            return "microjoint" in j.lower() and "microwave" not in j.lower()

        all_joints = {
            j: {
                **self.ontology_index[j],
                "position": [float(x) for x in self.joint_centroid[j]]
            }
            for j in self.sim_joints if not _is_bad(j)
        }
        object_snippet = self.movable_index
        digital_snippet = self.digital_index

        if not all_joints and not object_snippet:
            print(colored(" No joints or movable objects found in this scene", "red"))
            return None

        classify_messages = [
            {"role": "system", "content": (
                    "Classify the command and group relevant elements.\n"
                    "JOINTS:\n" + json.dumps(all_joints, indent=2) + "\n\n"
                                                                     "OBJECTS:\n" + json.dumps(object_snippet,
                                                                                               indent=2) + "\n\n"
                                                                                                           "DIGITAL_CONTROLS:\n" + json.dumps(
                digital_snippet, indent=2) + "\n\n"
                                             "SCENE_HINTS:\n" + json.dumps(self.scene_semantics, indent=2) + "\n\n"
                                                                                                             "Think step-by-step (CoT): 1) Rewrite command if vague (e.g., 'brew coffee' → 'press coffee button'). "
                                                                                                             "2) Group joints by type using names/categories/affordances (e.g., water_controls: those with 'sink'/'handle'; "
                                                                                                             "stove_knobs: 'stovetop'/'knob'; microwave_doors: 'microwave'/'door'). For stoves, subgroup by positions like front_left. "
                                                                                                             "3) Filter to command-relevant groups.\n"
                                                                                                             "Output ONLY raw JSON, no markdown or extra text: {'rewritten_command': str, 'groups': {'water_controls': [list], 'stove_knobs': [dict with subgroups], ...}}"
            )},
            {"role": "user", "content": f"Command: {command}"}
        ]
        try:
            classify_resp = client.chat.completions.create(model="gpt-4o", messages=classify_messages, temperature=0.1)
            txt = classify_resp.choices[0].message.content.strip()
            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0]
            classify_data = json.loads(txt)
            rewritten_command = classify_data['rewritten_command']
            groups = classify_data['groups']
        except Exception as e:
            print(colored(f" Classification failed: {e}", "red"))
            print("Raw LLM classification output:",
                  classify_resp.choices[0].message.content)
            return None

        messages = [
            {"role": "system", "content": (
                "You are an expert robot task planner.\n"
                "You receive:\n"
                "• JOINTS (continuous control)\n"
                "• OBJECTS (pick-and-place)\n"
                "• DIGITAL_CONTROLS (buttons)\n"
                "• SCENE_HINTS (coordinates)\n"
                "• GROUPS (pre-grouped by relevance)\n\n"
                f"JOINTS:\n{json.dumps(all_joints, indent=2)}\n\n"
                f"OBJECTS:\n{json.dumps(object_snippet, indent=2)}\n\n"
                f"DIGITAL_CONTROLS:\n{json.dumps(digital_snippet, indent=2)}\n\n"
                f"SCENE_HINTS:\n{json.dumps(self.scene_semantics, indent=2)}\n\n"
                f"GROUPS:\n{json.dumps(groups, indent=2)}\n\n"
                "Return ONLY valid JSON in this exact structure (no extra text or markdown):\n"
                "{\n"
                "  \"steps\": [\n"
                "    {\"action\": \"set_joint\", \"joint\": \"sim_name\", \"target\": value},\n"
                "    {\"action\": \"place\", \"object\": \"sim_body\", \"pose\": [x,y,z, qw,qx,qy,qz]},\n"
                "    {\"action\": \"press_button\", \"button\": \"sim_name\"},\n"
                "    {\"action\": \"spawn_object\", \"object_type\": \"food_type\", \"position\": [x,y,z]}\n"
                "  ],\n"
                "  \"success_criteria\": \"textual description\"\n"
                "}\n"
                "Example for 'open door': {\"steps\": [{\"action\": \"set_joint\", \"joint\": \"door_hinge\", \"target\": -1.57}], \"success_criteria\": \"Door is open\"}\n\n"
                "Guidelines (minimal):\n"
                "- Think step-by-step (CoT): 1) Match command to groups/affordances. 2) Select joints/objects using positions/categories. "
                "3) For opening/turning, use preferred_open/ranges. 4) Verify plan feasibility (e.g., radians, valid poses).\n"
                "- Use GROUPS for specifics (e.g., select from stove_knobs subgroups by spatial terms like 'front left').\n"
                "- All joint angles in radians. Prefer closest position for ambiguity.\n"
                "- The 'steps' must be an array of objects; each step must have 'action' and required fields. Use CoT to ensure this format.\n"
                "- Ensure 'steps' is always an array of action objects, even for simple tasks."
            )},
            {"role": "user", "content": f"Command: {rewritten_command}"}
        ]

        try:
            resp = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0.05, max_tokens=600)
            txt = resp.choices[0].message.content

            if "```json" in txt:
                txt = txt.split("```json")[1].split("```")[0]

            plan = json.loads(txt.strip())

            if not isinstance(plan.get("steps"), list):
                print("Invalid plan format detected; retrying with correction prompt...")
                correction_messages = messages + [
                    {"role": "assistant", "content": txt},
                    {"role": "user", "content": "Plan invalid. Correct it with CoT."}
                ]
                resp = client.chat.completions.create(model="gpt-4o", messages=correction_messages, temperature=0.1)
                corrected_txt = resp.choices[0].message.content.strip()
                if "```json" in corrected_txt:
                    corrected_txt = corrected_txt.split("```json")[1].split("```")[0]
                plan = json.loads(corrected_txt.strip())

            self.plan_cache[command] = plan
            with open(self.cache_path, 'w') as f:
                json.dump(self.plan_cache, f, indent=2)
            return plan
        except Exception as e:
            print(colored(f" Failed to parse LLM response: {e}", "red"))
            return None

    def execute_llm_plan(self, plan):
        if not plan or "steps" not in plan:
            print(colored(" Plan missing or has no 'steps' field", "red"))
            return False

        for step in plan["steps"]:
            action = step.get("action", "set_joint")

            if action == "set_joint":
                sim_name, target = step["joint"], step["target"]
                if sim_name not in self.sim_joints:
                    print(colored(f" Joint '{sim_name}' not found in simulation", "red"))
                    return False

                target = _deg2rad_if_needed(target)
                model = self.env.sim.model
                try:
                    jid = model.joint_name2id(sim_name)
                    lo_sim, hi_sim = model.jnt_range[jid]
                except Exception:
                    lo_sim, hi_sim = (0.0, 0.0)

                lo_ont, hi_ont = self.ontology_index.get(sim_name, {}).get("range", (-np.pi, np.pi))
                lo, hi = _effective_limits(lo_sim, hi_sim, (lo_ont, hi_ont))

                target = np.clip(target, lo, hi)

                if not self.move_joint_to_position(sim_name, target):
                    return False

            elif action == "pick":
                continue

            elif action == "place":
                obj = step["object"]
                pose = step["pose"]
                if not self.place_object(obj, pose):
                    return False

            elif action == "press_button":
                btn_name = step["button"]
                if not self.press_button(btn_name):
                    return False

            elif action == "spawn_object":
                object_type = step["object_type"]
                position = step["position"]
                spawned_obj = self.spawn_kitchen_object(object_type, position)
                if spawned_obj is None:
                    return False

            else:
                if "joint" in step and "target" in step:
                    sim_name, target = step["joint"], step["target"]
                    if sim_name not in self.sim_joints:
                        print(colored(f" Joint '{sim_name}' not found in simulation", "red"))
                        return False

                    target = _deg2rad_if_needed(target)

                    model = self.env.sim.model
                    try:
                        jid = model.joint_name2id(sim_name)
                        lo_sim, hi_sim = model.jnt_range[jid]
                    except Exception:
                        lo_sim, hi_sim = (0.0, 0.0)

                    lo_ont, hi_ont = self.ontology_index.get(sim_name, {}).get("range", (-np.pi, np.pi))
                    lo, hi = _effective_limits(lo_sim, hi_sim, (lo_ont, hi_ont))

                    target = np.clip(target, lo, hi)

                    if not self.move_joint_to_position(sim_name, target):
                        return False
                else:
                    print(colored(f" Unknown action: {action}", "red"))
                    return False
        return True

    def pick_object(self, object_name):
        try:
            # Get object body ID
            body_id = self.env.sim.model.body_name2id(object_name)

            # Get robot end-effector position
            robot_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("robot0_link7")]

            # Teleport object to gripper
            self.env.sim.data.body_xpos[body_id] = robot_pos + np.array([0, 0, 0.1])
            self.env.sim.forward()
            self.env.render()

            # print(colored(f" Picked up {object_name}", "green"))
            return True

        except Exception as e:
            print(colored(f" Failed to pick object '{object_name}': {e}", "red"))
            return False

    def place_object(self, object_name, pose):
        try:
            body_id = self.env.sim.model.body_name2id(object_name)
            pos, quat = np.asarray(pose[:3]), np.asarray(pose[3:7])

            self._teleport_body(body_id, pos, quat)

            self.env.sim.forward()
            self.env.render()

            print(colored(f" Placed {object_name} at {pos}", "green"))
            return True

        except Exception as e:
            print(colored(f" Failed to place object '{object_name}': {e}", "red"))
            return False

    def move_joint_to_position(self, joint_name, target_position):
        try:
            joint_id = self.sim_joints[joint_name]
            current_pos = self.env.sim.data.qpos[joint_id]

            steps = 20
            for step in range(steps):
                alpha = (step + 1) / steps
                new_pos = current_pos + alpha * (target_position - current_pos)

                self.env.sim.data.qpos[joint_id] = new_pos
                self.env.sim.forward()

                robot_action = np.zeros(self.env.action_dim)
                obs, reward, done, info = self.env.step(robot_action)
                self.env.render()

                time.sleep(0.05)

            final_pos = self.env.sim.data.qpos[joint_id]
            self.joint_positions[joint_name] = final_pos

            return True

        except Exception as e:
            print(colored(f" Failed to move joint '{joint_name}': {e}", "red"))
            return False

    def validate_task(self, plan, command):
        if "steps" in plan:
            errors = []
            for step in plan["steps"]:
                action = step.get("action", "set_joint")

                if action == "set_joint" or ("joint" in step and "target" in step):
                    if action == "set_joint":
                        sim_name = step["joint"]
                        target = step["target"]
                    else:
                        sim_name = step["joint"]
                        target = step["target"]

                    target = _deg2rad_if_needed(target)

                    actual = self.env.sim.data.qpos[self.sim_joints[sim_name]]
                    if abs(actual - target) > 5e-2:
                        errors.append(("joint", sim_name, target, float(actual)))

                elif action == "place":
                    obj = step["object"]
                    target_pose = step["pose"]
                    try:
                        body_id = self.env.sim.model.body_name2id(obj)
                        final_pos = self.env.sim.data.body_xpos[body_id]
                        target_pos = np.array(target_pose[:3])

                        if np.linalg.norm(final_pos - target_pos) > 0.03:
                            errors.append(("place", obj, target_pos.tolist(), final_pos.tolist()))
                    except Exception as e:
                        errors.append(("place", obj, "validation_failed", str(e)))

                elif action == "press_button":
                    # Button press validation
                    btn_name = step["button"]
                    try:
                        if "microwave" in btn_name:
                            mw = self.env.get_fixture(FixtureType.MICROWAVE)
                            state = mw.get_state()

                            if "start_button" in btn_name and not state.get("turned_on", False):
                                errors.append(("button", btn_name, "turned_on", "turned_off"))
                            elif "stop_button" in btn_name and state.get("turned_on", True):
                                errors.append(("button", btn_name, "turned_off", "turned_on"))
                        elif "coffee_machine" in btn_name:
                            cm = self.env.get_fixture("coffee_machine")
                            state = cm.get_state()["turned_on"]
                            if not state:
                                errors.append(("button", btn_name, "turned_on", "turned_off"))
                    except Exception as e:
                        errors.append(("button", btn_name, "validation_failed", str(e)))

                elif action == "spawn_object":
                    # Validation for spawned objects - check if they exist in the scene
                    object_type = step["object_type"]
                    position = step["position"]

            if not errors:
                # print(colored(" Auto-check passed – task complete", "green"))
                self.validate_with_vlm(command, plan)
                return

            diag_prompt = {
                "role": "user",
                "content": f"The following actions missed their targets: {errors}\n"
                           f"Original command: {plan.get('original_command', '')}"
            }
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[diag_prompt],
                temperature=0.3,
                max_tokens=200,
            )
            print(colored(" LLM diagnosis:\n" + resp.choices[0].message.content, "yellow"))
            self.validate_with_vlm(command, plan)

    def run_llm_driven_session(self):
        # print(colored(" LLM-Driven Task Execution System (MuJoCo-Range Auto-Augmented)", "cyan"))
        # print("=" * 65)

        possible_ontology_paths = [
            os.path.expanduser("~/PhD/digital twin/robocasa/onotolgy_final_full.json"),
            os.path.expanduser("~/robocasa/ontology/fonotolgy_final_full.json"),
            os.path.join(ROBOCASA_ROOT, "ontology/onotolgy_final_full.json"),
            os.path.join(ROBOCASA_ROOT, "onotolgy_final_full.json"),
            "onotolgy_final_full.json",
        ]

        ontology_loaded = False
        for ontology_path in possible_ontology_paths:
            if self.load_ontology(ontology_path):
                # print(f" Loaded ontology from: {ontology_path}")
                ontology_loaded = True
                break

        if not ontology_loaded:
            print(colored("  Warning: Could not load ontology file!", "yellow"))
            print("   The system will work with scene-detected joints only.")
            print("   Expected locations checked:")
            for path in possible_ontology_paths:
                print(f"     - {path}")
            # Don't return - continue without ontology
            self.ontology = {}

        self.build_joint_index()

        if not self.setup_environment():
            print(colored(" Failed to setup environment", "red"))
            return

        # Only print the list of movable objects
        if self.movable_index:
            # print("\nAvailable movable objects in the scene:")
            for obj_name in sorted(self.movable_index.keys()):
                # print(f"  - {obj_name}")
                pass
        else:
            print("No movable objects found in the scene.")

        if not self.sim_joints and not self.movable_index:
            print("The system will only work with joints defined in the ontology file or movable objects.")
        else:
            print(colored("  System ready for joint control and object manipulation!", "cyan"))

        # Clear problematic cached plans
        self.plan_cache.pop("open microwave", None)

        while True:
            try:
                command = input(colored(" Your command: ", "yellow")).strip()

                if command.lower() in ['quit', 'exit', 'q']:
                    break

                if not command:
                    continue

                print(colored(f"\n LLM is planning: '{command}'", "blue"))

                plan = self.ask_llm_for_plan(command)

                if plan:
                    plan["original_command"] = command

                    print(colored(f" Plan: {json.dumps(plan, indent=2)}", "blue"))

                    if self.execute_llm_plan(plan):
                        self.validate_task(plan, command)
                    else:
                        self.plan_cache.pop(command, None)
                        with open(self.cache_path, "w") as f:
                            json.dump(self.plan_cache, f, indent=2)
                        print(colored(" LLM plan execution failed\n", "red"))
                        print(colored("Plan failed – cache cleared for this command", "yellow"))
                else:
                    print(colored(" LLM could not create a plan (no ontology matches)\n", "red"))

            except KeyboardInterrupt:
                break

        self.env.close()
        print(colored("Environment closed", "green"))

def main():
    planner = LLMTaskPlanner()
    planner.run_llm_driven_session()

if __name__ == "__main__":
    main()
