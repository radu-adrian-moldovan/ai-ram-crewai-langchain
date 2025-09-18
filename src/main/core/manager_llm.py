"""
Centralized LLM Manager - Offline-First Implementation
Prioritizes local Ollama server calls, falls back to internet APIs only if local fails.
"""

import os
import logging
from typing import Any, Dict
from enum import Enum
import time

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers in priority order"""
    OLLAMA_LOCAL = "ollama_local"
    OPENAI_FALLBACK = "openai_fallback"

class ModelType(Enum):
    """Available model types"""
    VISION = "vision"  # For document processing with images
    TEXT = "text"     # For text-only processing

class LLMManager:
    """
    Centralized LLM Manager that ensures offline-first operation.
    
    Priority:
    1. Local Ollama server (localhost:11434) - OFFLINE
    2. OpenAI API (only if local fails) - ONLINE
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 default_timeout: int = 10):
        self.ollama_url = ollama_url
        self.default_timeout = default_timeout
        self.current_provider = None
        self.ollama_available = None  # Cache ollama availability
        
        # Model mappings for different providers
        self.model_mappings = {
            LLMProvider.OLLAMA_LOCAL: {
                ModelType.VISION: "llama3.2-vision:latest",
                ModelType.TEXT: "llama3.1:latest"
            },
            LLMProvider.OPENAI_FALLBACK: {
                ModelType.VISION: "gpt-4-vision-preview",
                ModelType.TEXT: "gpt-3.5-turbo"
            }
        }
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize and test available providers"""
        logger.info("🔍 Initializing LLM providers...")
        
        # Test Ollama availability first
        self.ollama_available = self._test_ollama_connection()
        
        if self.ollama_available:
            self.current_provider = LLMProvider.OLLAMA_LOCAL
            logger.info("✅ Using LOCAL Ollama server (OFFLINE)")
        else:
            # Check if OpenAI API key is available for fallback
            if os.getenv('OPENAI_API_KEY'):
                self.current_provider = LLMProvider.OPENAI_FALLBACK
                logger.warning("⚠️  Falling back to OpenAI API (ONLINE)")
            else:
                logger.error("❌ No LLM providers available!")
                raise RuntimeError("No LLM providers available. Install Ollama or set OPENAI_API_KEY")
    
    def _test_ollama_connection(self) -> bool:
        """Test if Ollama server is running and accessible"""
        try:
            import httpx
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    tags_data = response.json()
                    models = [model['name'] for model in tags_data.get('models', [])]
                    
                    # Check if required models are available
                    required_models = list(self.model_mappings[LLMProvider.OLLAMA_LOCAL].values())
                    available_models = []
                    
                    for required_model in required_models:
                        # Check if model exists (with or without :latest tag)
                        model_base = required_model.replace(':latest', '')
                        if any(model_base in model for model in models):
                            available_models.append(required_model)
                    
                    if available_models:
                        logger.info(f"✅ Ollama available with models: {available_models}")
                        return True
                    else:
                        logger.warning(f"⚠️  Ollama server running but required models not found")
                        logger.info(f"Available models: {models}")
                        logger.info(f"Required models: {required_models}")
                        return False
                        
        except Exception as e:
            logger.warning(f"⚠️  Ollama connection test failed: {e}")
            return False
        
        return False
    
    def get_llm_for_crewai(self, model_type: ModelType = ModelType.VISION) -> Any:
        """Get LLM instance for CrewAI framework"""
        if self.current_provider == LLMProvider.OLLAMA_LOCAL:
            return self._get_ollama_llm_crewai(model_type)
        else:
            return self._get_openai_llm_crewai(model_type)
    
    def get_llm_for_langchain(self, model_type: ModelType = ModelType.VISION) -> Any:
        """Get LLM instance for LangChain framework"""
        if self.current_provider == LLMProvider.OLLAMA_LOCAL:
            return self._get_ollama_llm_langchain(model_type)
        else:
            return self._get_openai_llm_langchain(model_type)
    
    def _get_ollama_llm_crewai(self, model_type: ModelType) -> Any:
        """Get Ollama LLM for CrewAI"""
        try:
            # Try using CrewAI's LLM class directly instead of LangChain's ChatOllama
            try:
                from crewai import LLM
                
                model_name = self.model_mappings[LLMProvider.OLLAMA_LOCAL][model_type]
                
                # CrewAI's LLM class with ollama provider
                llm = LLM(
                    model=f"ollama/{model_name}",
                    base_url=self.ollama_url,
                    temperature=0.1
                )
                
                logger.info(f"✅ CrewAI LLM ready: ollama/{model_name}")
                return llm
                
            except ImportError:
                # Fallback to LangChain ChatOllama with provider prefix
                from langchain_ollama import ChatOllama
                
                model_name = self.model_mappings[LLMProvider.OLLAMA_LOCAL][model_type]
                
                llm = ChatOllama(
                    base_url=self.ollama_url,
                    model=f"ollama/{model_name}",  # Add ollama provider prefix
                    temperature=0.1
                )
                
                logger.info(f"✅ CrewAI Ollama LLM ready: ollama/{model_name}")
                return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to create CrewAI Ollama LLM: {e}")
            # Force fallback to OpenAI
            self.current_provider = LLMProvider.OPENAI_FALLBACK
            return self._get_openai_llm_crewai(model_type)
    
    def _get_ollama_llm_langchain(self, model_type: ModelType) -> Any:
        """Get Ollama LLM for LangChain"""
        try:
            from langchain_ollama import ChatOllama
            
            model_name = self.model_mappings[LLMProvider.OLLAMA_LOCAL][model_type]
            
            llm = ChatOllama(
                base_url=self.ollama_url,
                model=model_name,
                temperature=0.1
            )
            
            logger.info(f"✅ LangChain Ollama LLM ready: {model_name}")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to create LangChain Ollama LLM: {e}")
            # Force fallback to OpenAI
            self.current_provider = LLMProvider.OPENAI_FALLBACK
            return self._get_openai_llm_langchain(model_type)
    
    def _get_openai_llm_crewai(self, model_type: ModelType) -> Any:
        """Get OpenAI LLM for CrewAI (fallback)"""
        try:
            from langchain_openai import ChatOpenAI
            
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError("OPENAI_API_KEY not found for fallback")
            
            model_name = self.model_mappings[LLMProvider.OPENAI_FALLBACK][model_type]
            
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.1,
                timeout=self.default_timeout
            )
            
            logger.warning(f"⚠️  CrewAI using OpenAI fallback: {model_name} (ONLINE)")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to create CrewAI OpenAI LLM: {e}")
            raise RuntimeError(f"No LLM available for CrewAI: {e}")
    
    def _get_openai_llm_langchain(self, model_type: ModelType) -> Any:
        """Get OpenAI LLM for LangChain (fallback)"""
        try:
            from langchain_openai import ChatOpenAI
            
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError("OPENAI_API_KEY not found for fallback")
            
            model_name = self.model_mappings[LLMProvider.OPENAI_FALLBACK][model_type]
            
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.1,
                timeout=self.default_timeout
            )
            
            logger.warning(f"⚠️  LangChain using OpenAI fallback: {model_name} (ONLINE)")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to create LangChain OpenAI LLM: {e}")
            raise RuntimeError(f"No LLM available for LangChain: {e}")
    
    async def test_llm_connection(self, framework: str = "test") -> Dict[str, Any]:
        """Test current LLM connection with a simple query"""
        try:
            if self.current_provider == LLMProvider.OLLAMA_LOCAL:
                llm = self.get_llm_for_langchain(ModelType.TEXT)
            else:
                llm = self._get_openai_llm_langchain(ModelType.TEXT)
            
            # Simple test query
            start_time = time.time()
            response = await llm.ainvoke("Say 'Hello' in one word.")
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'provider': self.current_provider.value if self.current_provider else 'none',
                'response_time': response_time,
                'response': str(response.content) if hasattr(response, 'content') else str(response),
                'framework': framework
            }
            
        except Exception as e:
            return {
                'success': False,
                'provider': self.current_provider.value if self.current_provider else 'none',
                'error': str(e),
                'framework': framework
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LLM manager status"""
        return {
            'current_provider': self.current_provider.value if self.current_provider else None,
            'ollama_available': self.ollama_available,
            'ollama_url': self.ollama_url,
            'openai_key_set': bool(os.getenv('OPENAI_API_KEY')),
            'offline_capable': self.ollama_available,
            'models': {
                'ollama': self.model_mappings.get(LLMProvider.OLLAMA_LOCAL, {}),
                'openai': self.model_mappings.get(LLMProvider.OPENAI_FALLBACK, {})
            }
        }
    
    def force_provider(self, provider: LLMProvider) -> bool:
        """Force switch to specific provider (for testing)"""
        try:
            if provider == LLMProvider.OLLAMA_LOCAL:
                if self._test_ollama_connection():
                    self.current_provider = provider
                    logger.info("🔄 Forced switch to Ollama LOCAL")
                    return True
                else:
                    logger.error("❌ Cannot force Ollama - not available")
                    return False
            elif provider == LLMProvider.OPENAI_FALLBACK:
                if os.getenv('OPENAI_API_KEY'):
                    self.current_provider = provider
                    logger.info("🔄 Forced switch to OpenAI FALLBACK")
                    return True
                else:
                    logger.error("❌ Cannot force OpenAI - API key not set")
                    return False
            return False
        except Exception as e:
            logger.error(f"❌ Failed to force provider switch: {e}")
            return False


# Global LLM manager instance
_llm_manager = None

def get_llm_manager() -> LLMManager:
    """Get global LLM manager instance (singleton)"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager

def reset_llm_manager():
    """Reset global LLM manager (for testing)"""
    global _llm_manager
    _llm_manager = None


# Convenience functions for direct usage
def get_crewai_llm(model_type: ModelType = ModelType.VISION) -> Any:
    """Get LLM for CrewAI framework"""
    return get_llm_manager().get_llm_for_crewai(model_type)

def get_langchain_llm(model_type: ModelType = ModelType.VISION) -> Any:
    """Get LLM for LangChain framework"""
    return get_llm_manager().get_llm_for_langchain(model_type)

async def test_llm_status() -> Dict[str, Any]:
    """Test current LLM status"""
    return await get_llm_manager().test_llm_connection()

def get_llm_status() -> Dict[str, Any]:
    """Get LLM manager status"""
    return get_llm_manager().get_status()
