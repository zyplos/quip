import type { PretrainedModelOptions } from "@huggingface/transformers";

export const SIGLIP_MODEL_NAME = "Xenova/siglip-base-patch16-224";
export const CLIP_MODEL_NAME = "Xenova/clip-vit-base-patch16";
export const JINA_V2_MODEL_NAME = "jinaai/jina-clip-v2";
export const JINA_V1_MODEL_NAME = "jinaai/jina-clip-v1";

export const MODEL_CONFIG: PretrainedModelOptions = {
  dtype: "fp16",
  // device: "dml",
  device: "cpu",
};

export const IMAGES_DIR = "./images";
// export const IMAGES_DIR = "./testImages";

export const EMBEDDING_LENGTH = 768;
