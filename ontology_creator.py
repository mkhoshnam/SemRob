import os
import json
import base64
import copy
import argparse
from pathlib import Path
from collections import defaultdict
import openai

# ---------- CONFIG ----------------------------------------------------------
RENDER_ROOT = Path.home() / "PhD" / "digital twin" / "robocasa"
BASIC_FILE = Path("ontology-extended/full_basic_robocasa_ontology.json")
FINAL_FILE = RENDER_ROOT / "final_ontology.json"
FINAL_FULL_FILE = RENDER_ROOT / "onotolgy_final_full.json"
MODEL = "gpt-4o"


OPENAI_KEY = "XXXX"
# ---------------------------------------------------------------------------
os.environ["OPENAI_API_KEY"] = OPENAI_KEY
client = openai.OpenAI()


# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--audit-stream", action="store_true",
                    help="Print a live VLM audit log during the verification pass")
ARGS = parser.parse_args()
AUDIT_STREAM = ARGS.audit_stream
# ---------------------------------------------------------------------------


client = openai.OpenAI()

# Map render-folder names > generic appliance type
APPLIANCE_MAPPING = {
    "renders_cabinets": "cabinet",
    "renders_microwaves_front_view2": "microwave",
    "renders_fridges": "fridge",
    "renders_coffee_machines": "coffee_machine",
    "renders_ovens": "oven",
    "renders_sink": "sink",
    "renders_stoves": "stove",
    "renders_stovetops": "stovetop",
}


def b64(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


# LOAD BASIC ONTOLOGY
def load_basic_ontology() -> dict:
    """Load the basic ontology structure"""
    if BASIC_FILE.exists():
        with BASIC_FILE.open(encoding='utf-8') as f:
            basic_ontology = json.load(f)
        print(f" Loaded basic ontology with {len(basic_ontology)} categories from: {BASIC_FILE}")
        return basic_ontology
    else:
        print(f"  Basic ontology not found at: {BASIC_FILE} – starting with empty structure")
        return {}


# SCAN THE RENDERS DIRECTORY
def find_render_folders() -> dict:
    """
    Return {appliance_type: [(instance_name, instance_folder_path, [img1, img2, ...]), ...]}
    """
    data = defaultdict(list)

    for folder_name, appliance in APPLIANCE_MAPPING.items():
        root = RENDER_ROOT / folder_name
        if not root.exists():
            print(f"[WARN] {root} missing")
            continue

        # Check for instance subdirectories
        subdirs = [p for p in root.iterdir() if p.is_dir()]

        if not subdirs:
            # If no subdirs, treat the folder as a single generic instance
            imgs = sorted([*root.glob("*.png"), *root.glob("*.jpg"), *root.glob("*.jpeg")])[:10]
            if imgs:
                data[appliance].append(("__generic__", root, imgs))
                print(f"Found {len(imgs)} images in {root} (generic instance)")
        else:
            # Process each instance subdirectory
            for inst_dir in subdirs:
                imgs = sorted([*inst_dir.glob("*.png"), *inst_dir.glob("*.jpg"), *inst_dir.glob("*.jpeg")])[:10]
                if imgs:
                    data[appliance].append((inst_dir.name, inst_dir, imgs))
                    print(f"Found {len(imgs)} images in {inst_dir} (instance: {inst_dir.name})")

    return data


# MERGE UTILITY
def create_basic_structure(appliance_type: str, instance_name: str) -> dict:
    """Create basic ontology structure for an appliance instance"""
    return {
        "class": appliance_type,
        "joints": [],
        "instances": [instance_name],
        "affordances": [],
        "ontology": {
            "object": {
                "description": f"A {appliance_type} instance: {instance_name}",
                "children": {}
            }
        }
    }


# VERIFY
ALLOWED_JOINT_TYPES = {"revolute", "prismatic", "fixed", "rotary", "digital"}

def _as_joint_dict(j):
    """Normalize any joint entry to a dict with name/type/sim_name."""
    if isinstance(j, str):
        return {"name": j, "type": None, "sim_name": None}
    if isinstance(j, dict):
        name = j.get("name") or j.get("joint") or j.get("sim_name")
        sim_name = j.get("sim_name") or j.get("joint_sim_name") or (name + "_sim" if name else None)
        jt = j.get("type")
        return {
            "name": name,
            "type": jt,
            "sim_name": sim_name,
            **{k: v for k, v in j.items() if k not in ("name", "type", "sim_name")}
        }
    return None

def _dedupe_joints_list(items):
    seen = set()
    out = []
    for it in items:
        if isinstance(it, dict) and it.get("name") and it.get("sim_name"):
            key = (it["name"], it["sim_name"])
        else:
            key = json.dumps(it, sort_keys=True)  # fallback
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out

def lint_ontology(draft: dict, appliance: str, instance: str):
    o = copy.deepcopy(draft)

    # normalize & dedupe joints
    joints = [_as_joint_dict(j) for j in o.get("joints", [])]
    joints = [j for j in joints if j and j.get("name")]  # drop invalid

    for j in joints:
        # map typographical types to canonical set
        if j.get("type") and j["type"] not in ALLOWED_JOINT_TYPES:
            mapping = {"hinge": "revolute", "slide": "prismatic", "rotational": "revolute"}
            j["type"] = mapping.get(j["type"], j["type"])

        # ensure sim_name exists
        if not j.get("sim_name"):
            j["sim_name"] = f"{j['name']}_sim"

    joints = _dedupe_joints_list(joints)
    o["joints"] = joints

    # check children & references
    issues = []
    children = o.get("ontology", {}).get("object", {}).get("children", {})
    if not isinstance(children, dict):
        children = {}
        o.setdefault("ontology", {}).setdefault("object", {})["children"] = children

    # remove obviously empty children
    to_delete = []
    for key, c in list(children.items()):
        if not isinstance(c, dict):
            issues.append({"path": f"ontology.object.children.{key}", "reason": "non-dict child", "fix": "drop"})
            to_delete.append(key)
            continue

        desc = (c.get("description") or "").strip()
        rel = c.get("related_tasks", [])
        has_mp = bool(c.get("manipulation_properties"))
        has_sim = bool(c.get("sim_name"))
        if not desc and not rel and not has_mp and not has_sim:
            issues.append({"path": f"ontology.object.children.{key}", "reason": "empty child", "fix": "drop"})
            to_delete.append(key)
            continue

        # walk manipulation_properties.*.joint references
        names = {j["name"] for j in joints}
        def walk(node, path):
            if isinstance(node, dict):
                for kk, vv in node.items():
                    if kk == "joint" and isinstance(vv, str):
                        if vv not in names:
                            issues.append({"path": path + ".joint", "reason": f"unknown joint '{vv}'", "fix": "rename or add joint"})
                    else:
                        walk(vv, path + f".{kk}")
            elif isinstance(node, list):
                for i, z in enumerate(node):
                    walk(z, path + f"[{i}]")
        walk(c.get("manipulation_properties", {}), f"ontology.object.children.{key}.manipulation_properties")

    for k in to_delete:
        del children[k]

    return o, issues

def verify_with_vlm(appliance: str, instance: str, images: list, sanitized: dict, model: str, client, audit_stream: bool=False):
    img_payloads = []
    for img_path in images[:3]:
        try:
            img_payloads.append({"type": "image_url", "image_url": {"url": b64(img_path)}})
        except Exception as e:
            print(f"[WARN] verify: failed to encode {img_path}: {e}")

    sys_prompt = {
        "role": "system",
        "content": (
            "You are an auditor of an automatically generated ontology for a household appliance.\n"
            "Perform explicit checks and report results **concisely** (no hidden reasoning):\n"
            "• Remove children that are not visible or implausible for this instance.\n"
            "• For each control (door/drawer/button/knob/slide), ensure a matching joint exists with a plausible type:\n"
            "  door→revolute, drawer→prismatic, button/switch→digital/fixed, knob→rotary/revolute.\n"
            "• Every joint MUST have nonempty 'name' and 'sim_name'.\n\n"
            "Return ONLY valid JSON in this schema:\n"
            "{\n"
            "  \"sanitized_ontology\": <ontology>,\n"
            "  \"issues\": [{\"path\": str, \"reason\": str, \"fix\": str}],\n"
            "  \"audit_log\": [\n"
            "    {\"check\": str, \"verdict\": \"pass\"|\"fail\", \"path\": str, \"evidence\": str, \"action\": str}\n"
            "  ]\n"
            "}\n"
            "Keep each audit_log entry short (one sentence)."
        )
    }

    user_content = [
        {"type": "text", "text": f"Appliance: {appliance}\nInstance: {instance}\nReview and correct the ontology below."},
        {"type": "text", "text": json.dumps(sanitized, ensure_ascii=False, indent=2)},
    ]
    user_content.extend(img_payloads)
    user_prompt = {"role": "user", "content": user_content}

    try:
        if audit_stream:
            print("\n--- VLM verification (live stream) ---")
            stream = client.chat.completions.create(
                model=model,
                messages=[sys_prompt, user_prompt],
                temperature=0.0,
                max_tokens=4000,
                stream=True,
            )
            chunks = []
            for chunk in stream:
                delta = None
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    pass
                if delta:
                    print(delta, end="", flush=True)
                    chunks.append(delta)
            print("")
            full_text = "".join(chunks)
        else:
            resp = client.chat.completions.create(
                model=model, messages=[sys_prompt, user_prompt], temperature=0.0, max_tokens=4000
            )
            full_text = resp.choices[0].message.content

        if "```json" in full_text:
            full_text = full_text.split("```json")[1].split("```")[0]
        elif "```" in full_text:
            full_text = full_text.split("```")[1].split("```")[0]

        obj = json.loads(full_text)
        if not isinstance(obj, dict):
            raise ValueError("Verifier returned non-dict JSON")

        # Defensive defaults
        obj.setdefault("sanitized_ontology", sanitized)
        obj.setdefault("issues", [])
        obj.setdefault("audit_log", [])
        return obj

    except Exception as e:
        print(f"[WARN] verify_with_vlm failed: {e}")
        return {
            "sanitized_ontology": sanitized,
            "issues": [{"path": "(global)", "reason": "verification_failed", "fix": "used_sanitized"}],
            "audit_log": [{"check": "verification", "verdict": "fail", "path": "(global)", "evidence": str(e), "action": "fallback"}],
        }

def _merge_lists_preserve_order(old_items, new_items, key=None):
    if key == "joints":
        # smart-dedupe joints by (name, sim_name)
        combined = []
        for src in (old_items, new_items):
            for it in src:
                combined.append(_as_joint_dict(it) if not (isinstance(it, dict) and "name" in it) else it)
        return _dedupe_joints_list(combined)
    # generic merge
    combined = old_items + [item for item in new_items if item not in old_items]
    return combined

def merge_category(old: dict, new: dict) -> dict:
    """Deep-merge 'new' into 'old' (keeps old keys that new lacks)"""
    merged = old.copy()
    for k, v in new.items():
        if k == "ontology":
            merged["ontology"] = merge_category(old.get("ontology", {}), v)
        elif isinstance(v, list):
            old_items = old.get(k, [])
            merged[k] = _merge_lists_preserve_order(old_items, v, key=k)  # ← NEW
        elif isinstance(v, dict):
            merged[k] = merge_category(old.get(k, {}), v)
        else:
            merged[k] = v
    return merged


def enrich_appliance_ontology(
        appliance: str,
        instance_name: str,
        images: list,
        basic_snippet: dict
) -> dict:
    """Send images and basic ontology to VLM for enrichment"""

    print(f"→ Enriching {appliance}/{instance_name} with {len(images)} images")

    img_payloads = []
    for img_path in images[:5]:
        try:
            img_payloads.append({
                "type": "image_url",
                "image_url": {"url": b64(img_path)}
            })
        except Exception as e:
            print(f"[WARN] Failed to encode image {img_path}: {e}")

    if not img_payloads:
        print(f"[ERROR] No valid images for {appliance}/{instance_name}")
        return create_basic_structure(appliance, instance_name)

    sys_prompt = {
        "role": "system",
        "content": (
            "You are an expert robotics ontology creator for Phase 2 NL-to-action systems. "
            "Based on the provided images and basic ontology structure, create a comprehensive "
            "instance-specific ontology that includes:\n\n"
            "1. All possible joints/moving parts (doors, drawers, handles, etc.)\n"
            "2. Joint types (revolute, prismatic, fixed)\n"
            "3. Simulator names for each joint (sim_name field)\n"
            "4. Object-level affordances (what this specific instance can do)\n"
            "5. Detailed descriptions and manipulation properties\n"
            "6. Grasp poses and state information\n\n"
            "EXTEND, do NOT replace, the base ontology structure provided.\n"
            "CRITICAL: Each joint MUST have a 'sim_name' field for simulator integration.\n\n"
            f"# BASE ONTOLOGY STRUCTURE\n"
            f"{json.dumps(basic_snippet, indent=2, ensure_ascii=False)}\n\n"
            "Return ONLY valid JSON in the same structure, but enriched with specific details "
            "for this instance."
        )
    }

    user_content = [
        {
            "type": "text",
            "text": (
                f"Appliance Category: {appliance}\n"
                f"Instance Name: {instance_name}\n"
                f"Create a detailed ontology for this specific {appliance} instance called '{instance_name}'. "
                f"Analyze the provided images to identify all movable parts, joints, and manipulation opportunities. "
                f"Include realistic sim_name values for each joint (like door_hinge, drawer_slide, handle_joint, etc.)."
            )
        }
    ]
    user_content.extend(img_payloads)

    user_prompt = {
        "role": "user",
        "content": user_content
    }

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[sys_prompt, user_prompt],
            temperature=0.2,
            max_tokens=4000,
        )

        response_content = resp.choices[0].message.content

        if "```json" in response_content:
            response_content = response_content.split("```json")[1].split("```")[0]
        elif "```" in response_content:
            response_content = response_content.split("```")[1].split("```")[0]

        enriched = json.loads(response_content.strip())

        enriched["instances"] = [instance_name]

        print(f" Successfully enriched {appliance}/{instance_name}")
        if enriched.get("joints"):
            print(f"  Found joints: {enriched['joints']}")
        if enriched.get("affordances"):
            print(f"  Object affordances: {enriched['affordances']}")

        return enriched

    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON response for {appliance}/{instance_name}: {e}")
        print(f"Raw response preview: {resp.choices[0].message.content[:500]}...")
        return create_basic_structure(appliance, instance_name)
    except Exception as e:
        print(f"[ERROR] API call failed for {appliance}/{instance_name}: {e}")
        return create_basic_structure(appliance, instance_name)


def main():
    print("=== Phase-1 Instance-Level Ontology Enrichment ===")
    print("=" * 55)

    basic_ontology = load_basic_ontology()

    final_ontology = basic_ontology.copy()

    if FINAL_FULL_FILE.exists():
        persisted = json.loads(FINAL_FULL_FILE.read_text(encoding='utf-8'))
        for cat, data in persisted.items():
            final_ontology[cat] = merge_category(final_ontology.get(cat, {}), data)
        print(f" Merged previous {FINAL_FULL_FILE.name} ({len(persisted)} categories)")
    elif FINAL_FILE.exists():
        # Fallback to old filename for compatibility
        persisted = json.loads(FINAL_FILE.read_text(encoding='utf-8'))
        for cat, data in persisted.items():
            final_ontology[cat] = merge_category(final_ontology.get(cat, {}), data)
        print(f" Merged previous {FINAL_FILE.name} ({len(persisted)} categories)")
    else:
        print("No existing final ontology – starting from basic ontology only.")

    print("\nScanning for render folders and instances...")
    appliance_instances = find_render_folders()

    if not appliance_instances:
        print("No render folders with PNG images found – aborting.")
        return

    total_instances = sum(len(instances) for instances in appliance_instances.values())
    print(f"\nFound {len(appliance_instances)} appliance types with {total_instances} total instances:")
    for appliance, instances in appliance_instances.items():
        instance_names = [name for name, _, _ in instances]
        print(f"  • {appliance}: {len(instances)} instances - {instance_names}")

    print(f"\nStarting instance-level enrichment...")
    processed_count = 0

    for appliance, instance_data in appliance_instances.items():
        print(f"\n{'=' * 50}")
        print(f"Processing appliance: {appliance.upper()}")
        print(f"{'=' * 50}")

        for instance_name, instance_folder, images in instance_data:
            processed_count += 1
            print(f"\n[{processed_count}/{total_instances}] Processing: {appliance}/{instance_name}")

            basic_snippet = final_ontology.get(appliance, create_basic_structure(appliance, instance_name))

            enriched_instance = enrich_appliance_ontology(
                appliance, instance_name, images, basic_snippet
            )

            draft = enriched_instance
            clean, lint_issues = lint_ontology(draft, appliance, instance_name)
            verified = verify_with_vlm(appliance, instance_name, images, clean, MODEL, client, audit_stream=AUDIT_STREAM)
            final_inst = verified.get("sanitized_ontology", clean)
            verify_issues = verified.get("issues", [])
            audit_log = verified.get("audit_log", [])

            if not AUDIT_STREAM:
                print("\n--- VLM verification audit ---")
                for i, entry in enumerate(audit_log[:50], 1):
                    check = entry.get("check", "")
                    verdict = entry.get("verdict", "")
                    path = entry.get("path", "")
                    evidence = entry.get("evidence", "")
                    action = entry.get("action", "")
                    print(f"  {i:02d}. [{verdict}] {check} @ {path} → {action}  | {evidence}")
                if len(audit_log) > 50:
                    print(f"  ... {len(audit_log) - 50} more entries omitted")

            # save per-instance files
            instance_ontology_file = instance_folder / "instance_ontology.json"  # ← renamed
            with instance_ontology_file.open('w', encoding='utf-8') as f:
                json.dump(final_inst, f, indent=2, ensure_ascii=False)

            issues_file = instance_folder / "instance_ontology_issues.json"
            with issues_file.open('w', encoding='utf-8') as f:
                json.dump({"lint": lint_issues, "verify": verify_issues}, f, indent=2, ensure_ascii=False)

            print(f" Saved instance ontology      → {instance_ontology_file}")
            print(f" Saved verification issues   → {issues_file}")

            # Merge verified instance into final ontology
            if appliance not in final_ontology:
                final_ontology[appliance] = create_basic_structure(appliance, instance_name)

            final_ontology[appliance] = merge_category(final_ontology[appliance], final_inst)

    # Save the comprehensive final ontology
    with FINAL_FULL_FILE.open('w', encoding='utf-8') as f:
        json.dump(final_ontology, f, indent=2, ensure_ascii=False)
    print(f"\n Saved comprehensive final ontology → {FINAL_FULL_FILE}")

    # print(f"\n ENRICHMENT SUMMARY:")
    # print(f"   Total appliance categories: {len(final_ontology)}")
    # print(f"   Total instances processed: {processed_count}")

    total_joints = 0
    total_affordances = 0
    for appliance, data in final_ontology.items():
        joints = len(data.get("joints", []))
        affordances = len(data.get("affordances", []))
        instances = len(data.get("instances", []))
        total_joints += joints
        total_affordances += affordances
        print(f"  • {appliance}: {instances} instances, {joints} joints, {affordances} affordances")

    print(f"   Total joints discovered: {total_joints}")
    print(f"   Total affordances identified: {total_affordances}")

    print(f"\n FILES GENERATED:")
    print(f"   Individual instance_ontology.json files in each instance folder")
    print(f"   Individual instance_ontology_issues.json files with lint/verify reports")
    print(f"   Comprehensive {FINAL_FULL_FILE.name} with all instances merged")


if __name__ == "__main__":
    main()
