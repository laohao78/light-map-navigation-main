import time
import logging
import io
import base64
from dataclasses import dataclass
from typing import Optional, List, Dict, Union, Tuple
from openai import OpenAI
from PIL import Image

@dataclass
class ChatResponse:
    """Data class for chat response"""
    content: Optional[str]
    messages: Optional[List[Dict]]

class APIClient:
    """Unified API client supporting both cloud and local models
    
    Features:
    - Support for cloud-based OpenAI models and local LLMs
    - Text and multimodal conversation capabilities
    - Single-turn and multi-turn dialogue support
    - Mixed text and image input support
    - Automatic retry mechanism
    - Comprehensive error handling
    """
    
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str = "gpt-3.5-turbo",
                 local_llm: bool = False,
                 max_retries: int = 3,
                 timeout: int = 30):
        """Initialize API client with configuration parameters"""
        self._setup_client(api_key, base_url, model_name, local_llm, max_retries, timeout)
        self._setup_logging()

    def _setup_client(self, api_key, base_url, model_name, local_llm, max_retries, timeout):
        """Setup client configuration"""
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.local_llm = local_llm
        self.max_retries = max_retries
        self.timeout = timeout
        
        self.client = OpenAI(
            api_key=api_key if not local_llm else "dummy-key",
            base_url=base_url,
            timeout=timeout
        )

    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)

    def encode_image(self, image_path: str) -> Optional[str]:
        """Encode image to base64 format"""
        try:
            with Image.open(image_path) as img:
                return self._process_image(img)
        except Exception as e:
            self.logger.error(f"Image encoding failed: {str(e)}")
            return None

    def _process_image(self, img: Image) -> str:
        """Process and encode image"""
        img = self._convert_image_mode(img)
        img = self._resize_image(img)
        return self._encode_to_base64(img)

    @staticmethod
    def _convert_image_mode(img: Image) -> Image:
        """Convert image to RGB mode"""
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            return background
        elif img.mode != 'RGB':
            return img.convert('RGB')
        return img

    @staticmethod
    def _resize_image(img: Image, max_size: int = 2048) -> Image:
        """Resize image if needed"""
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = tuple(int(dim * ratio) for dim in img.size)
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    @staticmethod
    def _encode_to_base64(img: Image) -> str:
        """Encode image to base64"""
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=95)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def _construct_messages(self,
                          prompt: str,
                          system_prompt: str = "",
                          image_data: Union[str, List[str]] = None,
                          conversation_context: List[Dict] = None) -> List[Dict]:
        """Construct message format for API requests"""
        if conversation_context:
            messages = conversation_context.copy()
            if image_data:
                # Construct message with images
                message = [{"type": "text", "text": prompt}]
                if isinstance(image_data, list):
                    # Multiple images
                    for img_data in image_data:
                        message.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                        })
                else:
                    # Single image
                    message.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                messages.append({"role": "user", "content": message})
            else:
                # Text-only message
                messages.append({"role": "user", "content": prompt})
        else:
            if image_data:
                # New conversation with images
                message = [{"type": "text", "text": prompt}]
                if isinstance(image_data, list):
                    for img_data in image_data:
                        message.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}
                        })
                else:
                    message.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                messages = [
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": message}
                ]
            else:
                # New text-only conversation
                messages = [
                    {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            
        return messages
    
    def chat(self,
             prompt: str,
             system_prompt: str = "",
             image_paths: Union[str, List[str]] = None,
             image_data: Union[str, List[str]] = None,
             conversation_context: List[Dict] = None,
             temperature: float = 0.3,
             stream: bool = False) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """Main chat interface for model interaction"""
        final_image_data = None
        
        # 处理 base64 图片数据
        if image_data:
            final_image_data = image_data if isinstance(image_data, list) else [image_data]
        # 处理图片路径
        elif image_paths:
            if isinstance(image_paths, list):
                final_image_data = []
                for img_path in image_paths:
                    img_data = self.encode_image(img_path)
                    if not img_data:
                        return None, None
                    final_image_data.append(img_data)
            else:
                final_image_data = self.encode_image(image_paths)
                if not final_image_data:
                    return None, None
                
        messages = self._construct_messages(
            prompt,
            system_prompt,
            final_image_data,
            conversation_context
        )
        
        for attempt in range(self.max_retries):
            try:
                if stream:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        stream=True
                    )
                    
                    # Handle streaming response
                    collected_messages = []
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            chunk_message = chunk.choices[0].delta.content
                            collected_messages.append(chunk_message)
                            print(chunk_message, end='', flush=True)
                            
                    print()  # New line
                    content = ''.join(collected_messages)
                    
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature
                    )
                    content = response.choices[0].message.content
                
                # Update conversation history
                messages.append({
                    "role": "assistant",
                    "content": content
                })
                
                return content, messages
                
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {str(e)}\n"
                    f"Error type: {type(e).__name__}\n"
                    f"Base URL: {self.base_url}"
                )
                if attempt == self.max_retries - 1:
                    self.logger.error("Maximum retry attempts reached")
                    return None, None
                time.sleep(2 ** attempt)
                
    def simple_chat(self, prompt: str, temperature: float = 0.3, stream: bool = False) -> Optional[str]:
        """Single-turn text conversation interface"""
        return self.chat(prompt, temperature=temperature, stream=stream)[0]
        
    def image_chat(self, prompt: str, image_path: str, temperature: float = 0.3) -> Optional[str]:
        """Single-turn image-based conversation interface"""
        return self.chat(prompt, image_paths=image_path, temperature=temperature)[0]
        
    def multi_turn_chat(self, 
                       prompt: str,
                       conversation_context: List[Dict],
                       temperature: float = 0.3,
                       stream: bool = False) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """Multi-turn conversation interface"""
        return self.chat(
            prompt=prompt, 
            conversation_context=conversation_context,
            temperature=temperature,
            stream=stream
        )

    def multi_modal_chat(self,
                        prompt: str,
                        image_paths: Union[str, List[str]],
                        conversation_context: Optional[List[Dict]] = None,
                        temperature: float = 0.3,
                        stream: bool = False) -> Tuple[Optional[str], Optional[List[Dict]]]:
        """
        Multi-turn conversation interface with mixed text and image input
        
        Args:
            prompt: User input text
            image_paths: Path(s) to image file(s)
            conversation_context: Conversation history
            temperature: Sampling temperature (0.0 to 2.0)
            stream: Whether to use streaming response
            
        Returns:
            Tuple of (response content, updated conversation history)
        """
        return self.chat(
            prompt=prompt,
            image_paths=image_paths,
            conversation_context=conversation_context,
            temperature=temperature,
            stream=stream
        )