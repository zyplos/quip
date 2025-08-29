import {
  AutoProcessor,
  RawImage,
  SiglipVisionModel,
} from "@huggingface/transformers";

import { MODEL_CONFIG, MODEL_NAME } from "./config";

const imageProcessor = await AutoProcessor.from_pretrained(MODEL_NAME);
const visionModel = await SiglipVisionModel.from_pretrained(
  MODEL_NAME,
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
