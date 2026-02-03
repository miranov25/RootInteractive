function is_little_endian(): boolean {
  let little_endian = true;
  const buffer = new ArrayBuffer(4);
  const uint32array = new Uint32Array(buffer);
  const uint8array = new Uint8Array(buffer);
  uint32array[0] = 0x0a0b0c0d;
    if (uint8array[0] === 0x0d && uint8array[1] === 0x0c && uint8array[2] === 0x0b && uint8array[3] === 0x0a) {
        little_endian = true;
    } else if (uint8array[0] === 0x0a && uint8array[1] === 0x0b && uint8array[2] === 0x0c && uint8array[3] === 0x0d) {
        little_endian = false;
    }   
    return little_endian;
}

export const BYTE_ORDER = is_little_endian() ? "little" : "big";

function swap16(buffer: ArrayBuffer) {
  const x = new Uint8Array(buffer);
  for (let i = 0, end = x.length; i < end; i += 2) {
      const t = x[i];
      x[i] = x[i + 1];
      x[i + 1] = t;
  }
}
function swap32(buffer: ArrayBuffer) {
  const x = new Uint8Array(buffer);
  for (let i = 0, end = x.length; i < end; i += 4) {
      let t = x[i];
      x[i] = x[i + 3];
      x[i + 3] = t;
      t = x[i + 1];
      x[i + 1] = x[i + 2];
      x[i + 2] = t;
  }
}
function swap64(buffer: ArrayBuffer) {
  const x = new Uint8Array(buffer);
  for (let i = 0, end = x.length; i < end; i += 8) {
      let t = x[i];
      x[i] = x[i + 7];
      x[i + 7] = t;
      t = x[i + 1];
      x[i + 1] = x[i + 6];
      x[i + 6] = t;
      t = x[i + 2];
      x[i + 2] = x[i + 5];
      x[i + 5] = t;
      t = x[i + 3];
      x[i + 3] = x[i + 4];
      x[i + 4] = t;
  }
}

export function swap(buffer: ArrayBuffer, dtype: string) {
  switch (dtype) {
      case "uint16":
      case "int16":
          swap16(buffer);
          break;
      case "uint32":
      case "int32":
      case "float32":
          swap32(buffer);
          break;
      case "float64":
          swap64(buffer);
          break;
  }
}

function hash32(x: number) {
  x |= 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

function noiseSigned(seed: number, i: number) {
  // [-0.5, 0.5)
  return (hash32(seed ^ Math.imul(i, 0x9e3779b9)) / 2**32) - 0.5;
}

export function seedFromString(s: string) {
  let h = 2166136261; // FNV-1a base
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function decodeFixedPoint(value: number, scale: number, origin: number) {
  return origin + scale * value;
}

function decodeFixedPointDithered(value: number, scale: number, origin: number, seed: number, index: number) {
  return origin + scale * (value + noiseSigned(seed, index));
}

export function decodeFixedPointArray(array: number[], scale: number, origin: number, dither: boolean = false, seedString: string = "default") {
  const seed = seedFromString(seedString);
  const decodedArray = new Float64Array(array.length);
    for (let i = 0; i < array.length; i++) {
        if (dither) {
            decodedArray[i] = decodeFixedPointDithered(array[i], scale, origin, seed, i);
        } else {
            decodedArray[i] = decodeFixedPoint(array[i], scale, origin);
        }
    }
    return decodedArray;
}