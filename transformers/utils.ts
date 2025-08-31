import { Tensor } from "@huggingface/transformers";

export interface CompareData {
  [model: string]: {
    similarity: number;
    dimension: number;
  };
}

/*
valid types
from tensor.js in "@huggingface/transformers"
export const DataTypeMap = Object.freeze({
    float32: Float32Array,
    // Limited availability of Float16Array across browsers:
    // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Float16Array
    float16: typeof Float16Array !== "undefined" ? Float16Array: Uint16Array,
    float64: Float64Array,
    string: Array, // string[]
    int8: Int8Array,
    uint8: Uint8Array,
    int16: Int16Array,
    uint16: Uint16Array,
    int32: Int32Array,
    uint32: Uint32Array,
    int64: BigInt64Array,
    uint64: BigUint64Array,
    bool: Uint8Array,
    uint4: Uint8Array,
    int4: Int8Array,
});
*/
type TensorTypes = "float32" | "int64";
interface TensorTypeMap {
  float32: Float32Array;
  int64: BigInt64Array;
}
export function createEmptyTensor<T extends TensorTypes>(
  type: T,
  dims: number[]
) {
  const size = dims.reduce((a, b) => a * b);
  let emptyData: TensorTypeMap[T];

  if (type === "float32") {
    emptyData = new Float32Array(size) as TensorTypeMap[T];
  } else if (type === "int64") {
    emptyData = new BigInt64Array(size) as TensorTypeMap[T];
  } else {
    throw new Error(`EmptyTensor function doesn't make ${type} arrays`);
  }

  return new Tensor(type, emptyData, dims);
}

// thanks gemini 2.5 flash
export function markdownTable(data: CompareData): string {
  // Check if the data is empty. If so, return an empty string.
  if (Object.keys(data).length === 0) {
    return "";
  }

  // Define the table headers.
  const headers = ["Model", "Similarity", "Dimension"];

  // Create the Markdown header row.
  const headerRow = `| ${headers.join(" | ")} |`;

  // Create the Markdown separator row for alignment.
  const separatorRow = "|---|---|---|";

  // Map over the keys of the data object to create each data row.
  const dataRows = Object.keys(data).map((model) => {
    // biome-ignore lint/style/noNonNullAssertion: shut up
    const { similarity, dimension } = data[model]!;
    // Format the row with the model name, similarity, and dimension.
    return `| ${model} | ${similarity} | ${dimension} |`;
  });

  // Join all parts (header, separator, and data rows) with newlines.
  return [headerRow, separatorRow, ...dataRows].join("\n");
}
