import {
  AutoProcessor,
  RawImage,
  CLIPVisionModelWithProjection,
} from "@huggingface/transformers";

import { MODEL_CONFIG, JINA_V1_MODEL_NAME } from "@/transformers/config";

const imageProcessor = await AutoProcessor.from_pretrained(
  "Xenova/clip-vit-base-patch32"
);
const visionModel = await CLIPVisionModelWithProjection.from_pretrained(
  JINA_V1_MODEL_NAME,
  MODEL_CONFIG
);

export default async function generateImageEmbedding(
  path: string
): Promise<number[]> {
  const image = await RawImage.read(path);
  const imageInputs = await imageProcessor(image);
  const { image_embeds } = await visionModel(imageInputs);

  return image_embeds[0].data;
}
