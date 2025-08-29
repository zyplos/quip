import {
  AutoProcessor,
  RawImage,
  CLIPVisionModel,
} from "@huggingface/transformers";

import { MODEL_CONFIG, CLIP_MODEL_NAME } from "@/transformers/config";

const imageProcessor = await AutoProcessor.from_pretrained(CLIP_MODEL_NAME);
const visionModel = await CLIPVisionModel.from_pretrained(
  CLIP_MODEL_NAME,
  MODEL_CONFIG
);

export default async function generateImageEmbedding(
  path: string
): Promise<number[]> {
  const image = await RawImage.read(path);
  const imageInputs = await imageProcessor(image);
  const { image_embeds } = await visionModel(imageInputs);

  return image_embeds.normalize().tolist()[0];
}
