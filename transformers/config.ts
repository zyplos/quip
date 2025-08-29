import type { PretrainedModelOptions } from "@huggingface/transformers";

export const MODEL_NAME = "Xenova/siglip-base-patch16-224";

export const MODEL_CONFIG: PretrainedModelOptions = {
  dtype: "fp16",
  // device: "dml",
  device: "cpu",
};

export const IMAGES_DIR = "./images";
// export const IMAGES_DIR = "./testImages";

export const EMBEDDING_LENGTH = 768;
