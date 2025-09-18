"""
Configuration utility for loading and validating agent prompts
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PromptConfigLoader:
    """Utility class for loading and validating agent prompt configurations."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "prompts.yaml"
        self.config_path = Path(config_path)
        self.config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load agent configurations from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"✓ Loaded agent configurations from {self.config_path}")
            self._validate_config()
            return self.config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _validate_config(self):
        """Validate the loaded configuration structure."""
        if not self.config:
            raise ValueError("Configuration is empty")
        
        # Check required sections
        required_sections = ["agents", "tasks", "langchain_prompts"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate agents section
        agents = self.config["agents"]
        if not isinstance(agents, list) or len(agents) == 0:
            raise ValueError("Agents section must be a non-empty list")
        
        required_agent_fields = ["name", "role", "goal", "backstory", "tools"]
        for agent in agents:
            for field in required_agent_fields:
                if field not in agent:
                    raise ValueError(f"Agent missing required field: {field}")
        
        logger.info(f"✓ Configuration validation passed - {len(agents)} agents loaded")
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """Get configuration for a specific agent."""
        if not self.config:
            self.load_config()
        
        for agent_cfg in self.config["agents"]:
            if agent_cfg["name"] == agent_name:
                return agent_cfg
        raise ValueError(f"Agent configuration not found: {agent_name}")
    
    def get_task_config(self, task_name: str) -> Dict[str, str]:
        """Get task configuration for a specific task."""
        if not self.config:
            self.load_config()
        
        if task_name not in self.config["tasks"]:
            raise ValueError(f"Task configuration not found: {task_name}")
        
        return self.config["tasks"][task_name]
    
    def get_langchain_prompt(self, prompt_name: str) -> str:
        """Get LangChain prompt for a specific operation."""
        if not self.config:
            self.load_config()
        
        if prompt_name not in self.config["langchain_prompts"]:
            raise ValueError(f"LangChain prompt not found: {prompt_name}")
        
        return self.config["langchain_prompts"][prompt_name]
    
    def list_agents(self) -> List[str]:
        """List all available agent names."""
        if not self.config:
            self.load_config()
        
        return [agent["name"] for agent in self.config["agents"]]
    
    def list_tasks(self) -> List[str]:
        """List all available task names."""
        if not self.config:
            self.load_config()
        
        return list(self.config["tasks"].keys())

# Convenience function for quick access
def load_prompt_config(config_path: str = None) -> Dict[str, Any]:
    """Load prompt configuration from YAML file."""
    loader = PromptConfigLoader(config_path)
    return loader.load_config()

# Example usage and validation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Test loading configuration
        loader = PromptConfigLoader()
        config = loader.load_config()
        
        print(f"Configuration loaded successfully!")
        print(f"Available agents: {loader.list_agents()}")
        print(f"Available tasks: {loader.list_tasks()}")
        
        # Test agent retrieval
        for agent_name in loader.list_agents():
            agent_cfg = loader.get_agent_config(agent_name)
            print(f"\nAgent: {agent_name}")
            print(f"  Role: {agent_cfg['role']}")
            print(f"  Goal: {agent_cfg['goal'][:50]}...")
            print(f"  Tools: {agent_cfg['tools']}")
        
        # Test task retrieval
        for task_name in loader.list_tasks():
            task_cfg = loader.get_task_config(task_name)
            print(f"\nTask: {task_name}")
            print(f"  Description: {task_cfg['description'][:50]}...")
            print(f"  Expected Output: {task_cfg['expected_output'][:50]}...")
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        exit(1)
