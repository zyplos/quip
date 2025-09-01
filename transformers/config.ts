import type { PretrainedModelOptions } from "@huggingface/transformers";

export const MODEL_NAME = "jinaai/jina-clip-v1";

export const MODEL_CONFIG: PretrainedModelOptions = {
  dtype: "fp16",
  device: "dml",
  // device: "cpu",
};

export const IMAGES_DIR = "./images";
// export const IMAGES_DIR = "./testImages";

export const EMBEDDING_LENGTH = 768;
