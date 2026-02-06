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

export function decodeFixedPointArray(array: number[], scale: number, origin: number, sentinels: any = {}, dither: boolean = false, seedString: string = "default") {
  const seed = seedFromString(seedString);
  const decodedArray = new Float64Array(array.length);
    for (let i = 0; i < array.length; i++) {
        if(sentinels.nan != null && array[i] === sentinels.nan){
          decodedArray[i] = NaN
          continue
        }
        if(sentinels.posinf != null && array[i] === sentinels.posinf){
          decodedArray[i] = Infinity
          continue
        }
        if(sentinels.neginf != null && array[i] === sentinels.neginf){
          decodedArray[i] = -Infinity
          continue
        }
        if (dither) {
            decodedArray[i] = decodeFixedPointDithered(array[i], scale, origin, seed, i);
        } else {
            decodedArray[i] = decodeFixedPoint(array[i], scale, origin);
        }
    }
    return decodedArray;
}

export function to_fixed_point(x: number, scale: number, origin: number): number {
  return Math.round((x - origin) / scale);
}

export function decodeSinhArray(array: number[], mu: number, sigma0: number, sigma1: number, sentinels: any = {}, dither: boolean = false, seedString: string = "default") {
  const seed = seedFromString(seedString);
  const decodedArray = new Float64Array(array.length);
  const sigmaRatio = sigma0 / sigma1;
    for (let i = 0; i < array.length; i++) {
        if(sentinels.nan != null && array[i] === sentinels.nan){
          decodedArray[i] = NaN
          continue
        }
        if(sentinels.posinf != null && array[i] === sentinels.posinf){
          decodedArray[i] = Infinity
          continue
        }
        if(sentinels.neginf != null && array[i] === sentinels.neginf){
          decodedArray[i] = -Infinity
          continue
        }
        if (dither) {
            decodedArray[i] = sigmaRatio * Math.sinh(sigma0 * (array[i] + noiseSigned(seed, i)) + mu);
        } else {
            decodedArray[i] = sigmaRatio * Math.sinh(sigma0 * array[i] + mu);
        }
    }
    return decodedArray;
}

export function quantizeSinhArray(array: number[], sigma0: number, sigma1: number, nBits: number){
  if(sigma0 <= 0 || sigma1 <= 0){
    throw "Sigma cannot be negative";
  }
  const invSigma0 = 1/sigma0;
  const sigmaRatio = sigma1*invSigma0;
  const encodedArray = new Int32Array(array.length);
  const nanSentinel = -(2**(nBits-1))
  const posinfSentinel = 2**(nBits-1)-1
  const neginfSentinel = -(2**(nBits-1))+1
  for(let i=0; i < array.length; i++){
    let quantized = Math.round(Math.asinh(array[i]*sigmaRatio)*invSigma0)
    quantized = quantized < neginfSentinel ? neginfSentinel : quantized;
    quantized = quantized > posinfSentinel ? posinfSentinel : quantized;
    quantized = isNaN(quantized) ? nanSentinel : quantized;
    encodedArray[i] = quantized;
  }
  return {"array": encodedArray, "sentinels": {"nan": nanSentinel, "posinf":posinfSentinel, "neginf":neginfSentinel}}
}

export function decodeArray(arrayIn: any, instructions: any, env: any){
  const inflate = env.builtins.inflate
  const enableDithering = env.enableDithering
  const seed = env.seed

  let arrayOut = arrayIn

  for(let i=instructions.length-1; i>=0; i--){
      const action = Object.prototype.toString.call(instructions[i]) === '[object String]' ? instructions[i] : instructions[i][0]
      const actionParams = Object.prototype.toString.call(instructions[i]) === '[object String]' ? null : instructions[i][1]

      if (action == "base64_decode"){
        const s = atob(arrayOut)
        arrayOut = new Uint8Array(s.length)
        for (let j = 0; j < s.length; j++) { 
          arrayOut[j] = s.charCodeAt(j)
        }
      }
      if (action == "inflate") {
        arrayOut = inflate(arrayOut)
      }
      if(action == "array"){
        const dtype = actionParams
        const ab = arrayOut.buffer.slice(arrayOut.byteOffset, arrayOut.byteOffset + arrayOut.byteLength)
        if(env.byteorder !== BYTE_ORDER){
          swap(ab.buffer, dtype)
        }
        if (dtype == "int8"){
          arrayOut = new Int8Array(ab)
        }
        if (dtype == "int16"){
          arrayOut = new Int16Array(ab)
        }
        if (dtype == "uint16"){
          arrayOut = new Uint16Array(ab)
        }
        if (dtype == "int32"){
          arrayOut = new Int32Array(ab)
        }
        if (dtype == "uint32"){
          arrayOut = new Uint32Array(ab)
        }
        if (dtype == "float32"){
          arrayOut = new Float32Array(ab)
          arrayOut = new Float64Array(arrayOut)
        }
        if (dtype == "float64"){
          arrayOut = new Float64Array(ab)
        }
      }
      if (action == "code") {
        let size = arrayOut.length
        let arrayOutNew = new Array(arrayOut.length)
        for (let j = 0; j < size; j++) {
          arrayOutNew[j] = actionParams.valueCode[arrayOut[j]]
        }
        arrayOut=arrayOutNew
      }
      if (action == "linear") {
        const dither = actionParams.dither == "true" || (enableDithering && (actionParams.dither === "toggle" || actionParams.dither == null))
        arrayOut = decodeFixedPointArray(Array.from(arrayOut) as number[], actionParams.scale, actionParams.origin, actionParams.sentinels || {}, dither, seed)
      }
      if (action == "sinh"){
        const dither = actionParams.dither == "true" || (enableDithering && (actionParams.dither === "toggle" || actionParams.dither == null))
        arrayOut = decodeSinhArray(Array.from(arrayOut) as number[], actionParams.mu, actionParams.sigma0, actionParams.sigma1, actionParams.sentinels || {}, dither, seed)
      }
  }
  return arrayOut
}
  