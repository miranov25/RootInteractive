import { pathToFileURL } from "url";
import path from "path";
import zlib from "zlib";
import fs from "fs"

const tempdir = process.argv[2];

const file = path.join(tempdir, "SerializationUtils.js");

const SerializationUtils = await import(pathToFileURL(file).href);

const inputPath = process.argv[3];
const data = JSON.parse(fs.readFileSync(inputPath, "utf8"));

function allclose(A,B,abs_tol,rel_tol,nan_equal=false){
    if(A === B) return true
    if(!A || !B || A.length !== B.length) return false
    for(let i=0; i<A.length; i++){
        const a = A[i];
        const b = B[i];
        if(Number.isNaN(a) || Number.isNaN(b)){
            if(!(nan_equal && Number.isNaN(a) && Number.isNaN(b))){
                return false;
            }
        } else {
            const diff = Math.abs(a - b);
            if (diff > abs_tol && diff > rel_tol * Math.max(Math.abs(a), Math.abs(b))) {
                return false;
            }
        }
    }
    return true;
}

for(let i=0; i<data.length; i++){
    const {data_ref, data_compressed, pipeline, meta} = data[i]
    const binary_string_ref = Buffer.from(data_ref, 'base64');
    const dtype = meta.dtype
    let decoded_ref = null
    if(dtype === "float64"){
        if (binary_string_ref.length % 8 !== 0) {
            throw new Error(`Invalid float64 payload length ${binary_string_ref.length} (not divisible by 8)`);
        }
        decoded_ref = new Float64Array(binary_string_ref.buffer, binary_string_ref.byteOffset, binary_string_ref.length / Float64Array.BYTES_PER_ELEMENT)
    } else if(dtype === "int32"){
        if (binary_string_ref.length % 4 !== 0) {
            throw new Error(`Invalid float64 payload length ${binary_string_ref.length} (not divisible by 4)`);
        }
        decoded_ref = new Int32Array(binary_string_ref.buffer, binary_string_ref.byteOffset, binary_string_ref.length / Int32Array.BYTES_PER_ELEMENT)
    } else {
        throw new Error("Reference deserialization only implemented for float64 and int32")
    }
    const env = {
        builtins: {
            "inflate": zlib.inflateSync
        },
        seed: meta.name,
        enableDithering: meta.enableDithering,
        byteorder: meta.byteorder
    }
    const decoded_array = SerializationUtils.decodeArray(data_compressed, pipeline, env)
    if(!allclose(decoded_array, decoded_ref, meta.abs_tol, meta.rel_tol, meta.nan_equal)){
        throw new Error(`${meta.name} test failed. Expected ${decoded_ref} but got ${decoded_array}`)
    }
    console.log(`Test case passed: ${meta.name}`)
}

if(!data.length){
    console.log("No tests found")
}
