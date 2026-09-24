"""Vision Encoder with PEFT Adapters."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from peft import get_peft_model, LoraConfig
from typing import Optional, Dict, List, Tuple
from pathlib import Path


# Your local model paths
HF_MODELS_DIR = "/mnt/lustre-grete/projects/intern_agc_emmy/hf_models"

# Available vision models in HF format
VISION_MODELS = {
    'dinov2-small': f"{HF_MODELS_DIR}/dinov3/facebook_dinov2-small",
    'dinov3-vits16': f"{HF_MODELS_DIR}/dinov3/dinov3_vits16",
    'dinov3-vitb16': f"{HF_MODELS_DIR}/dinov3/dinov3_vitb16",
    # Aliases for convenience
    'dinov3-small': f"{HF_MODELS_DIR}/dinov3/dinov3_vits16",
    'dinov3-base': f"{HF_MODELS_DIR}/dinov3/dinov3_vitb16",
}


class VisionEncoder(nn.Module):
    """Vision encoder for sign language video frames."""
    
    def __init__(
        self,
        model_name: str = 'dinov2-small',
        use_peft: bool = True,
        peft_config: Optional[Dict] = None,
        out_dim: int = 1024,
        freeze_base: bool = True,
    ):
        """Initialize vision encoder."""
        super().__init__()
        
        if peft_config is None:
            peft_config = {'rank': 8, 'alpha': 16, 'dropout': 0.1}
        
        # Get model path
        if model_name not in VISION_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(VISION_MODELS.keys())}")
        
        model_path = VISION_MODELS[model_name]
        
        print(f"Loading vision encoder: {model_name}")
        print(f"Path: {model_path}")
        
        # Load base model from local directory
        self.base_model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,  # No internet needed!
        )
        
        print(f"✅ Loaded base model")
        print(f"   Parameters: {sum(p.numel() for p in self.base_model.parameters()):,}")
        
        # Get hidden dimension
        # DinoV2-small: 384, DinoV2-base: 768, DinoV2-large: 1024
        if hasattr(self.base_model.config, 'hidden_size'):
            num_features = self.base_model.config.hidden_size
        else:
            # Fallback for models without config.hidden_size
            num_features = 384  # DinoV2-small default
        
        print(f"   Hidden dimension: {num_features}")
        
        if use_peft:
            print("\nApplying PEFT LoRA (matching old Sign2GPT)...")
            
            # Auto-detect target modules based on model type
            # DinoV2 uses: query, key, value, dense (attention) + fc1, fc2 (MLP)
            # DinoV3 uses: q_proj, k_proj, v_proj, o_proj (attention) + up_proj, down_proj (MLP)
            
            # We use regex target_modules to achieve layer-specific LoRA
            layers_to_transform = peft_config.get('layers_to_transform', [9, 10, 11])
            layer_pattern = '|'.join(str(l) for l in layers_to_transform)
            
            if peft_config.get('target_modules') is not None:
                # User explicitly specified target modules (could be regex or list)
                target_modules = peft_config['target_modules']
                print(f"  Using user-specified target_modules: {target_modules}")
            else:
                # Auto-detect based on model name and build regex for layer-specific LoRA
                model_type = type(self.base_model).__name__
                if 'dinov3' in model_name.lower() or 'DINOv3' in model_type:
                    # DinoV3: layer.X.attention.{q,k,v,o}_proj + layer.X.mlp.{up,down}_proj
                    target_modules = rf'layer\.({layer_pattern})\.(attention\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(up_proj|down_proj))'
                    print(f"  Detected DinoV3 model, applying LoRA to layers {layers_to_transform}")
                else:
                    # DinoV2: encoder.layer.X.attention.attention.{query,key,value} + encoder.layer.X.attention.output.dense + encoder.layer.X.mlp.{fc1,fc2}
                    target_modules = rf'encoder\.layer\.({layer_pattern})\.(attention\.attention\.(query|key|value)|attention\.output\.dense|mlp\.(fc1|fc2))'
                    print(f"  Detected DinoV2 model, applying LoRA to layers {layers_to_transform}")
            
            lora_config = LoraConfig(
                r=peft_config['rank'],
                lora_alpha=peft_config['alpha'],
                target_modules=target_modules,
                lora_dropout=peft_config.get('dropout', 0.1),
                bias="none",
                init_lora_weights="gaussian",  # Use Gaussian init (closer to old Sign2GPT)
            )
            
            self.base_model = get_peft_model(self.base_model, lora_config)
            
            self._reinit_lora_weights(std=0.02)
            
            self.base_model.print_trainable_parameters()
        
        elif freeze_base:
            # Freeze base model if not using PEFT
            print("\nFreezing base model...")
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.projection = nn.Linear(num_features, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)  # BatchNorm1d like old Sign2GPT
        
        # Store output dimension for downstream use
        self.out_dim = out_dim  
        
        print(f"\n✅ Vision encoder created!")
        print(f"   Input: (B, T, 3, H, W) video frames")
        print(f"   Output: (B, T, {out_dim}) features")
    
    def _reinit_lora_weights(self, std: float = 0.02):
        """Reinitialize LoRA weights to match old Sign2GPT initialization."""
        for name, param in self.base_model.named_parameters():
            if 'lora_A' in name:
                nn.init.normal_(param, mean=0.0, std=std)
            elif 'lora_B' in name:
                nn.init.zeros_(param)
        print(f"  Reinitialized LoRA weights: A=normal(0, {std}), B=zeros")
    
    def forward(
        self,
        frames: torch.Tensor,
        return_cls_token: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through vision encoder."""
        # Reshape if needed: (B, T, 3, H, W) -> (B*T, 3, H, W)
        if frames.ndim == 5:
            batch_size, seq_len = frames.shape[:2]
            frames = frames.reshape(-1, *frames.shape[2:])
        else:
            batch_size, seq_len = None, None
        
        # Extract features
        outputs = self.base_model(frames)
        
        # HuggingFace models: last_hidden_state[:, 0] is CLS token
        cls_features = outputs.last_hidden_state[:, 0]  # (B*T, hidden_dim)
        
        projected = self.projection(cls_features)  # (B*T, out_dim)
        projected = self.norm(projected)  # BatchNorm1d expects (N, C)
        
        # Reshape back if we had batch dimension
        if batch_size is not None and seq_len is not None:
            projected = projected.reshape(batch_size, seq_len, -1)
            cls_features = cls_features.reshape(batch_size, seq_len, -1)
        
        if return_cls_token:
            return projected, cls_features
        else:
            return projected, None


def create_vision_encoder(
    model_name: str = 'dinov2-small',
    use_peft: bool = True,
    peft_config: Optional[Dict] = None,
    out_dim: int = 1024,
    device: str = "cuda",
) -> VisionEncoder:
    """Create vision encoder with PEFT adapters from local models."""
    if peft_config is None:
        peft_config = {'rank': 8, 'alpha': 16, 'dropout': 0.1}
    
    # Create model
    model = VisionEncoder(
        model_name=model_name,
        use_peft=use_peft,
        peft_config=peft_config,
        out_dim=out_dim,
    )
    
    # Move to device
    if device is not None and torch.cuda.is_available():
        model = model.to(device)
        print(f"\n✅ Model moved to {device}")
    
    return model


def load_image_processor(model_name: str = 'dinov2-small'):
    """Load image processor for vision model."""
    if model_name not in VISION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(VISION_MODELS.keys())}")
    
    model_path = VISION_MODELS[model_name]
    
    processor = AutoImageProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    
    print(f"✅ Loaded image processor for {model_name}")
    
    return processor


# Test code
if __name__ == "__main__":
    print("="*60)
    print("Testing Vision Encoder")
    print("="*60)
    print()
    
    # Check available models
    print("Available models:")
    for name, path in VISION_MODELS.items():
        exists = Path(path).exists()
        print(f"  {'✅' if exists else '❌'} {name}: {path}")
    print()
    
    # Test 1: Create vision encoder with PEFT
    print("="*60)
    print("Test 1: Vision Encoder with PEFT LoRA")
    print("="*60)
    
    try:
        model = create_vision_encoder(
            model_name='dinov2-small',
            use_peft=True,
            peft_config={'rank': 8, 'alpha': 16, 'dropout': 0.1},
            out_dim=1024,
            device='cpu',  # Use CPU for testing
        )
        
        print("\n✅ Model created successfully!")
        
        # Test forward pass
        print("\nTesting forward pass...")
        
        # Test with 4D input: (B*T, 3, H, W)
        dummy_input_4d = torch.randn(4, 3, 224, 224)  # 4 frames
        features_4d, cls_4d = model(dummy_input_4d, return_cls_token=True)
        
        print(f"✅ 4D Input: {dummy_input_4d.shape}")
        print(f"   Output features: {features_4d.shape}")
        print(f"   Output CLS tokens: {cls_4d.shape if cls_4d is not None else None}")
        
        # Test with 5D input: (B, T, 3, H, W)
        dummy_input_5d = torch.randn(2, 10, 3, 224, 224)  # 2 videos, 10 frames each
        features_5d, cls_5d = model(dummy_input_5d, return_cls_token=True)
        
        print(f"\n✅ 5D Input: {dummy_input_5d.shape}")
        print(f"   Output features: {features_5d.shape}")
        print(f"   Output CLS tokens: {cls_5d.shape if cls_5d is not None else None}")
        
        # Check trainable parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        
        print(f"\n📊 Parameter count:")
        print(f"   Trainable: {trainable:,}")
        print(f"   Total: {total:,}")
        print(f"   Trainable%: {100*trainable/total:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Load image processor
    print("\n" + "="*60)
    print("Test 2: Image Processor")
    print("="*60)
    
    try:
        processor = load_image_processor('dinov2-small')
        
        print(f"✅ Processor loaded")
        print(f"   Type: {type(processor)}")
        
        # Test processing
        dummy_image = torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8)
        # Convert to PIL for processor
        from PIL import Image
        import numpy as np
        pil_image = Image.fromarray(dummy_image.numpy())
        
        inputs = processor(images=pil_image, return_tensors="pt")
        
        print(f"✅ Processed image")
        print(f"   Input shape: {inputs['pixel_values'].shape}")
        print(f"   Input dtype: {inputs['pixel_values'].dtype}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Test without PEFT (full fine-tuning)
    print("\n" + "="*60)
    print("Test 3: Vision Encoder without PEFT (Full Fine-tuning)")
    print("="*60)
    
    try:
        model_full = create_vision_encoder(
            model_name='dinov2-small',
            use_peft=False,
            out_dim=512,  # Different output dim
            device='cpu',
        )
        
        print("\n✅ Model created (full fine-tuning mode)")
        
        trainable = sum(p.numel() for p in model_full.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model_full.parameters())
        
        print(f"\n📊 Parameter count:")
        print(f"   Trainable: {trainable:,}")
        print(f"   Total: {total:,}")
        print(f"   Trainable%: {100*trainable/total:.2f}%")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\nYou can now use the vision encoder in your training pipeline:")
    print("""
from src.models.vision_encoder import create_vision_encoder

# Create encoder
encoder = create_vision_encoder(
    model_name='dinov2-small',
    use_peft=True,
    peft_config={'rank': 8, 'alpha': 16},
    out_dim=1024,
)

# Process video frames
frames = torch.randn(batch_size, num_frames, 3, 224, 224)
features, cls_tokens = encoder(frames)

# features: (batch_size, num_frames, out_dim)
# Use these features for your downstream task
""")
