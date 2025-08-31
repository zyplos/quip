import {
  AutoProcessor,
  RawImage,
  SiglipModel,
} from "@huggingface/transformers";

import { MODEL_CONFIG, SIGLIP_MODEL_NAME } from "@/transformers/config";
import { createEmptyTensor } from "../utils";

const imageProcessor = await AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME);
const model = await SiglipModel.from_pretrained(
  SIGLIP_MODEL_NAME,
  MODEL_CONFIG
);

const emptyTextInputs = {
  input_ids: createEmptyTensor("int64", [2, 64]),
};

export default async function generateImageEmbedding(
  path: string
): Promise<number[]> {
  const image = await RawImage.read(path);
  const imageInputs = await imageProcessor(image);
  const { image_embeds } = await model({ ...emptyTextInputs, ...imageInputs });

  return image_embeds.normalize().tolist()[0];
}
