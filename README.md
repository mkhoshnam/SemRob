# SemRob
*A framework for semantic memory and ontology-driven robotic manipulation*

SemRob enables **natural-language command execution** for household robotics.  
You can issue commands such as:

- `"open drawer"`
- `"turn on the microwave"`
- `"press the coffee machine button"` or even `"I need coffee"` (the LLM reasons about the intent)
- Multi-step tasks like:  
  `"open microwave, then turn on the left bottom stove, then…"`

### How it works
- **LLM** → Performs task planning based on a semantic ontology  
- **Digital Twin** → Executes the plan in simulation (e.g., RoboCasa, CALVIN)  
- **VLM** → Validates whether the intended task was successfully completed  

Originally developed for **RoboCasa**, but easily adaptable to other environment — tested also on **CALVIN**.

