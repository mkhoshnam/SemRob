import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any


class GenericRoboCasaOntologyGenerator:
    def __init__(self):
        self.robocasa_path = Path("/media/mohammad/5f7c6d23-fd63-41c6-a822-d02e7c729060/robocasa")
        self.fixtures_path = self.robocasa_path / "robocasa" / "models" / "assets" / "fixtures"
        self.objects_path = self.robocasa_path / "robocasa" / "models" / "assets" / "objects" / "objaverse"

        # Create ontology-extended folder in current directory
        self.output_dir = Path("./ontology-extended")
        self.output_dir.mkdir(exist_ok=True)

        # Target fixture categories with their generic names
        self.fixture_categories = {
            "cabinets": "cabinet",
            "coffee_machines": "coffee_machine",
            "counters": "counter",
            "dishwashers": "dishwasher",
            "fridges": "fridge",
            "hoods": "hood",
            "microwaves": "microwave",
            "ovens": "oven",
            "sinks": "sink",
            "stoves": "stove",
            "stovetops": "stovetop",
            "toasters": "toaster"
        }

        self.object_categories = {
            "alcohol": "alcohol", "apple": "apple", "avocado": "avocado", "bagel": "bagel",
            "bagged_food": "bagged_food", "baguette": "baguette", "banana": "banana", "bar": "bar",
            "bar_soap": "bar_soap", "beer": "beer", "bell_pepper": "bell_pepper", "bottled_drink": "bottled_drink",
            "bottled_water": "bottled_water", "bowl": "bowl", "boxed_drink": "boxed_drink", "boxed_food": "boxed_food",
            "bread": "bread", "broccoli": "broccoli", "cake": "cake", "can": "can", "candle": "candle",
            "canned_food": "canned_food", "carrot": "carrot", "cereal": "cereal", "cheese": "cheese",
            "chips": "chips", "chocolate": "chocolate", "coffee_cup": "coffee_cup", "condiment": "condiment",
            "corn": "corn", "croissant": "croissant", "cucumber": "cucumber", "cup": "cup",
            "cutting_board": "cutting_board", "donut": "donut", "egg": "egg", "eggplant": "eggplant",
            "fish": "fish", "fork": "fork", "garlic": "garlic", "hot_dog": "hot_dog", "jam": "jam",
            "jug": "jug", "ketchup": "ketchup", "kettle": "kettle", "kiwi": "kiwi", "knife": "knife",
            "ladle": "ladle", "lemon": "lemon", "lime": "lime", "mango": "mango", "milk": "milk",
            "mug": "mug", "mushroom": "mushroom", "onion": "onion", "orange": "orange", "pan": "pan",
            "peach": "peach", "pear": "pear", "plate": "plate", "potato": "potato", "rolling_pin": "rolling_pin",
            "scissors": "scissors", "shaker": "shaker", "soap_dispenser": "soap_dispenser", "spatula": "spatula",
            "sponge": "sponge", "spoon": "spoon", "spray": "spray", "squash": "squash", "steak": "steak",
            "sweet_potato": "sweet_potato", "tangerine": "tangerine", "teapot": "teapot", "tomato": "tomato",
            "tray": "tray", "waffle": "waffle", "water_bottle": "water_bottle", "wine": "wine", "yogurt": "yogurt"
        }

    def validate_paths(self) -> bool:
        """Validate that the RoboCasa path exists and has the expected structure."""
        if not self.fixtures_path.exists():
            print(f"Error: Fixtures path not found: {self.fixtures_path}")
            return False

        missing_categories = []
        for category in self.fixture_categories.keys():
            category_path = self.fixtures_path / category
            if not category_path.exists():
                missing_categories.append(category)

        if missing_categories:
            print(f"Warning: Missing fixture categories: {missing_categories}")

        return True

    def parse_mjcf_file(self, xml_file: Path) -> List[str]:
        """
        Parse an MJCF XML file and extract joint names.

        Args:
            xml_file: Path to the model.xml file

        Returns:
            List of joint names found in the model
        """
        joints = []

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Find all joint elements in the XML
            for joint in root.iter('joint'):
                joint_name = joint.get('name')
                if joint_name:
                    joints.append(joint_name)

        except ET.ParseError as e:
            print(f"Error parsing {xml_file}: {e}")
        except Exception as e:
            print(f"Unexpected error parsing {xml_file}: {e}")

        return joints

    def create_joint_ontology(self, joint_name: str) -> Dict[str, Any]:
        """
        Create the basic ontology structure for a joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Dictionary containing the joint ontology structure
        """
        return {
            "description": "",
            "state": "N/A",
            "related_tasks": [],
            "effects": {},
            "children": {}
        }

    def create_generic_ontology(self, generic_name: str, joints: List[str], instances: List[str]) -> Dict[str, Any]:
        """
        Create the generic ontology structure for an appliance category.

        Args:
            generic_name: Generic name (e.g., "cabinet", "fridge")
            joints: List of all unique joint names in this category
            instances: List of model instances that contributed to this category

        Returns:
            Dictionary containing the complete generic ontology
        """
        ontology = {
            "class": generic_name,
            "joints": sorted(joints),  # Sort for consistency
            "instances": instances,  # Keep track of contributing models
            "ontology": {
                "object": {
                    "children": {}
                }
            }
        }

        # Add each joint to the ontology children
        for joint_name in joints:
            ontology["ontology"]["object"]["children"][joint_name] = self.create_joint_ontology(joint_name)

        return ontology

    def process_fixture_category(self, category: str, generic_name: str) -> Dict[str, Any]:
        """
        Process all models in a fixture category and create a single generic entry.

        Args:
            category: Fixture category name (e.g., 'microwaves', 'cabinets')
            generic_name: Generic name for this category (e.g., 'microwave', 'cabinet')

        Returns:
            Dictionary containing the generic ontology for this category
        """
        category_path = self.fixtures_path / category

        if not category_path.exists():
            print(f"Skipping missing category: {category}")
            return {}

        print(f"Processing category: {category} -> {generic_name}")

        all_joints = set()  # Use set to avoid duplicates
        instances = []

        if category == "cabinets":
            for xml_file in category_path.glob("*.xml"):
                if xml_file.name.endswith("_ontology.json"):  # Skip any ontology files
                    continue

                model_name = xml_file.stem  # filename without .xml extension
                print(f"  Processing cabinet model: {model_name}")

                # Parse the MJCF file and extract joints
                joints = self.parse_mjcf_file(xml_file)
                all_joints.update(joints)
                instances.append(model_name)

                if joints:
                    print(f"    Found joints: {joints}")
                else:
                    print(f"    No joints found")
        else:
            # Standard handling for other categories - look for model.xml in subdirectories
            for model_dir in category_path.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = model_dir.name
                model_xml = model_dir / "model.xml"
                instances.append(model_name)

                if not model_xml.exists():
                    print(f"  Warning: No model.xml found in {model_dir}")
                    continue

                print(f"  Processing model: {model_name}")

                # Parse the MJCF file and extract joints
                joints = self.parse_mjcf_file(model_xml)
                all_joints.update(joints)

                if joints:
                    print(f"    Found joints: {joints}")
                else:
                    print(f"    No joints found")

        # Create single generic ontology for this category
        if instances:
            generic_ontology = self.create_generic_ontology(generic_name, list(all_joints), instances)
            return {generic_name: generic_ontology}
        else:
            return {}

    def process_object_category(self, category: str, generic_name: str) -> Dict[str, Any]:
        category_path = self.objects_path / category

        if not category_path.exists():
            return {}

        instances = []
        for model_dir in category_path.iterdir():
            if model_dir.is_dir():
                instances.append(model_dir.name)

        if instances:
            ontology = {
                "class": generic_name,
                "instances": instances,
                "ontology": {
                    "object": {
                        "children": {}
                    }
                }
            }
            return {generic_name: ontology}
        else:
            return {}

    def generate_ontology_file(self, generic_name: str, ontology_data: Dict[str, Any]) -> None:
        """
        Generate and save the ontology JSON file for a generic category.

        Args:
            generic_name: Generic category name (e.g., "cabinet", "fridge")
            ontology_data: Ontology data for this generic category
        """
        # Save to JSON file with generic name
        output_file = self.output_dir / f"{generic_name}_ontology.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ontology_data, f, indent=2, ensure_ascii=False)

        print(f"Generated ontology file: {output_file}")

        actual_data = ontology_data[generic_name]
        print(f"  Generic name: {generic_name}")
        print(f"  Total unique joints: {len(actual_data.get('joints', []))}")
        print(f"  Contributing models: {len(actual_data['instances'])}")
        if actual_data.get('joints'):
            print(f"  Joints: {actual_data['joints']}")

    def generate_full_ontology_file(self, all_ontologies: Dict[str, Dict[str, Any]]) -> None:
        """
        Generate a combined ontology file containing all generic categories.

        Args:
            all_ontologies: Dictionary with all generic category ontologies
        """
        # Save to JSON file
        output_file = self.output_dir / "full_basic_robocasa_ontology.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_ontologies, f, indent=2, ensure_ascii=False)

        print(f"\n Generated FULL generic ontology file: {output_file}")

        # Calculate totals - fixed the statistics calculation
        total_categories = len(all_ontologies)
        total_instances = sum(
            len(category_data["instances"])
            for category_data in all_ontologies.values()
        )
        total_joints = sum(
            len(category_data.get("joints", []))
            for category_data in all_ontologies.values()
        )

        # print(f"   Total generic categories: {total_categories}")
        # print(f"   Total model instances processed: {total_instances}")
        # print(f"   Total unique joints across all categories: {total_joints}")

    def generate_all_ontologies(self) -> None:
        """Generate generic ontology files for all fixture categories."""
        # print("Starting Generic RoboCasa ontology generation...")
        # print(f"RoboCasa path: {self.robocasa_path}")
        # print(f"Fixtures path: {self.fixtures_path}")
        # print(f"Objects path: {self.objects_path}")
        # print(f"Output directory: {self.output_dir}")
        # print()

        if not self.validate_paths():
            print("Path validation failed. Please check the RoboCasa path.")
            return

        total_categories = 0
        all_ontologies = {}

        for category, generic_name in self.fixture_categories.items():
            print(f"\n{'=' * 50}")
            category_ontology = self.process_fixture_category(category, generic_name)

            if category_ontology:
                self.generate_ontology_file(generic_name, category_ontology)
                all_ontologies.update(category_ontology)
                total_categories += 1
            else:
                print(f"No models found for category: {category}")

        for category, generic_name in self.object_categories.items():
            category_ontology = self.process_object_category(category, generic_name)
            if category_ontology:
                all_ontologies.update(category_ontology)

        if all_ontologies:
            self.generate_full_ontology_file(all_ontologies)

        print(f"\n{'=' * 50}")
        print(" Generic ontology generation complete!")
        print(f" Individual generic files: {total_categories}")
        print(f" Output files saved to: {self.output_dir}")
        print(f"\n Generated files:")

        for generic_name in self.fixture_categories.values():
            if generic_name in all_ontologies:
                print(f"  • {generic_name}_ontology.json")
        if all_ontologies:
            print(f"  • full_robocasa_ontology.json (combined)")


def main():
    """Main function - just run this in PyCharm."""
    print("Generic RoboCasa Ontology Generator")
    print("=" * 50)

    try:
        generator = GenericRoboCasaOntologyGenerator()
        generator.generate_all_ontologies()
        print("\n Script completed successfully!")

    except KeyboardInterrupt:
        print("\n Operation cancelled by user")

    except Exception as e:
        print(f"\n Error during ontology generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
