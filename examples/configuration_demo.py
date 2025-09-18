"""
Example demonstrating how both CrewAI and LangChain processors use the same configuration
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from main.core.manager_propmt import PromptConfigLoader

def demonstrate_configuration_usage():
    """Demonstrate how both processors would use the same configuration."""
    
    # Load configuration
    loader = PromptConfigLoader()
    config = loader.load_config()
    
    print("=== Agent Configuration Demonstration ===\n")
    
    # Show available agents
    print(f"Available agents: {loader.list_agents()}")
    print(f"Available tasks: {loader.list_tasks()}")
    print()
    
    # Demonstrate how CrewAI would use the configuration
    print("=== CrewAI Agent Setup Example ===")
    for agent_name in loader.list_agents():
        agent_cfg = loader.get_agent_config(agent_name)
        print(f"\nAgent: {agent_name}")
        print(f"Role: {agent_cfg['role']}")
        print(f"Goal: {agent_cfg['goal']}")
        print(f"Tools: {agent_cfg['tools']}")
        print(f"Backstory preview: {agent_cfg['backstory'][:100]}...")
        
        # This is how CrewAI would create the agent:
        # agent = Agent(
        #     role=agent_cfg["role"],
        #     goal=agent_cfg["goal"],
        #     backstory=agent_cfg["backstory"],
        #     llm=self.llm,
        #     verbose=True,
        #     allow_delegation=False
        # )
    
    print("\n" + "="*60)
    
    # Demonstrate how LangChain would use the configuration
    print("\n=== LangChain Agent Setup Example ===")
    for agent_name in loader.list_agents():
        agent_cfg = loader.get_agent_config(agent_name)
        prompt_key = {
            "document_classification_agent": "classification",
            "field_extraction_agent": "extraction", 
            "accuracy_assessment_agent": "accuracy_assessment",
            "final_decision_agent": "final_decision"
        }.get(agent_name, "classification")
        
        langchain_prompt = loader.get_langchain_prompt(prompt_key)
        
        print(f"\nAgent: {agent_name}")
        print(f"System Message: {agent_cfg['backstory'][:100]}...")
        print(f"Human Prompt: {langchain_prompt}")
        
        # This is how LangChain would create the agent:
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", agent_cfg["backstory"]),
        #     ("human", langchain_prompt),
        #     MessagesPlaceholder(variable_name="agent_scratchpad")
        # ])
    
    print("\n" + "="*60)
    
    # Demonstrate task configuration usage
    print("\n=== Task Configuration Example ===")
    for task_name in loader.list_tasks():
        task_cfg = loader.get_task_config(task_name)
        print(f"\nTask: {task_name}")
        print(f"Description Template: {task_cfg['description']}")
        print(f"Expected Output: {task_cfg['expected_output']}")
        
        # This is how CrewAI would create the task:
        # task = Task(
        #     description=task_cfg["description"].format(image_path="example.jpg", ...),
        #     agent=agent,
        #     expected_output=task_cfg["expected_output"]
        # )
    
    print("\n" + "="*60)
    print("\n✓ Both processors use identical configuration from prompts.yaml!")
    print("✓ No duplication of agent definitions or prompts!")
    print("✓ Easy to maintain and update all agent configurations in one place!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_configuration_usage()
