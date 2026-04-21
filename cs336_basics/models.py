from pydantic import BaseModel


class LLModelConfig(BaseModel):
    d_model: int
    max_seq_len: int


class RoPEConfig(BaseModel):
    theta: float
    d_k: int


class MHSAConfig(BaseModel):
    rope_config: RoPEConfig | None = None
    num_heads: int
    causal: bool = True


class RMSConfig(BaseModel):
    eps: float = 1e-5


class GLUConfig(BaseModel):
    d_ff: int | None = None


class TransformerBlockConfig(BaseModel):
    mhsa: MHSAConfig
    rms: RMSConfig
    glu: GLUConfig
