import {
  AutoProcessor,
  RawImage,
  SiglipVisionModel,
} from "@huggingface/transformers";

import { MODEL_CONFIG, SIGLIP_MODEL_NAME } from "@/transformers/config";

const imageProcessor = await AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME);
const visionModel = await SiglipVisionModel.from_pretrained(
  SIGLIP_MODEL_NAME,
  MODEL_CONFIG
);

export default async function generateImageEmbedding(
  path: string
): Promise<number[]> {
  const image = await RawImage.read(path);
  const imageInputs = await imageProcessor(image);
  const { pooler_output } = await visionModel(imageInputs);

  return pooler_output.normalize().tolist()[0];
}
