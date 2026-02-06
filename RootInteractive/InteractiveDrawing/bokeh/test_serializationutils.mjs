import { pathToFileURL } from "url";
import path from "path";
import zlib from "zlib";

const tempdir = process.argv[2];

const file = path.join(tempdir, "SerializationUtils.js");

const SerializationUtils = await import(pathToFileURL(file).href);

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

function allclose_abs(A,B,abs_tol){
    if(A === B) return true
    if(!A || !B || A.length !== B.length) return false
    for(let i=0; i<A.length; i++){
        const a = A[i];
        const b = B[i];
        if(Math.abs(a - b) > abs_tol){
            return false;
        }
    }
    return true;
}

function nan_equal(A,B){
    if(A === B) return true
    if(!A || !B || A.length !== B.length) return false
    for(let i=0; i<A.length; i++){
        const a = A[i];
        const b = B[i];
        if(Number.isNaN(a) || Number.isNaN(b)){
            if(!(Number.isNaN(a) && Number.isNaN(b))){
                return false;
            }
        } else {
            if (a !== b) {
                return false;
            }
        }
    }
    return true;
}

function test_fixedToFloat64Array(){
    const fixed_array = new Int32Array([1000, 2000, -3000, 4000]);
    const scale = 0.01;
    const offset = -10.0;
    const expected = new Float64Array([0.0, 10.0, -40.0, 30.0]);
    const result = SerializationUtils.decodeFixedPointArray(fixed_array, scale, offset);
    if(!allclose(result, expected, 1e-10, 1e-10)){
        throw new Error(`fixedToFloat64Array test failed. Expected ${expected} but got ${result}`);
    }
}

function test_fixedToFloat64Array_no_math_random(){
    const fixed_array = new Int32Array([1000, 2000, -3000, 4000]);
    const scale = 0.01;
    const offset = 5.0;
    const expected = new Float64Array([15.0, 25.0, -25.0, 45.0]);
    const original = Math.random;
    Math.random = () => { throw new Error("Math.random called"); };
    try {
        const result = SerializationUtils.decodeFixedPointArray(fixed_array, scale, offset, {}, true, "test-seed");
        if(!allclose_abs(result, expected, .005 + 1e-8)){
            throw new Error(`fixedToFloat64Array no math random test failed. Expected ${expected} but got ${result}`);
        }
    } finally {
        Math.random = original;
    }
}

function test_sinhToFloat64Array(){
    const array_orig = new Float64Array([15.0, 25.0, NaN, 0, -1e6, -1, Infinity, -Infinity]);
    const sigma0 = 1e-3;
    const sigma1 = 10;
    const quantized = SerializationUtils.quantizeSinhArray(array_orig, sigma0, sigma1, 16);
    const array_new = SerializationUtils.decodeSinhArray(quantized.array, 0, sigma0, sigma1, quantized.sentinels);
    if(!allclose(array_orig, array_new, sigma0 * (sigma0 / sigma1) / 2, sigma0 / 2, true)){
        throw new Error(`sinhToFloat64Array test failed. Expected ${array_orig} but got ${array_new}`)
    }
}

function test_sinhToFloat64Array_dither(){
    const array_orig = new Float64Array([15.0, 25.0, NaN, 0, -1e6, -1, Infinity, -Infinity]);
    const sigma0 = 1e-3;
    const sigma1 = 10;
    const quantized = SerializationUtils.quantizeSinhArray(array_orig, sigma0, sigma1, 16);
    const original = Math.random
    Math.random = () => { throw new Error("Math.random called"); };
    try {
        const array_new = SerializationUtils.decodeSinhArray(quantized.array, 0, sigma0, sigma1, quantized.sentinels, true, "test-seed");
        if(!allclose(array_orig, array_new, sigma0 * (sigma0 / sigma1), sigma0, true)){
            throw new Error(`sinhToFloat64Array_dither test failed. Expected ${array_orig} but got ${array_new}, abs_tol=${sigma0 * (sigma0 / sigma1)}, rel_tol=${sigma0}`)
        }
    } finally {
        Math.random = original;
    }
}

function test_simpleRoundtripLossless(){
    const array_orig = new Float64Array([15.0, 25.0, NaN, 0, -1e6, -1, Infinity, -Infinity]);
    const pipeline = [["array","float64"], ["inflate"], ["base64_decode"]];
    const compressed = zlib.deflateSync(array_orig).toString('base64');
    const env = {
        "builtins": {
            "inflate": zlib.inflateSync
        },
        "byteorder": SerializationUtils.BYTE_ORDER,
        "enableDithering": false
    }
    const array_new = SerializationUtils.decodeArray(compressed, pipeline, env)
    if(!nan_equal(array_new, array_orig)){
        throw new Error(`simpleRoundtripLossless test failed. Expected ${array_orig} but got ${array_new}`)
    }
}

test_fixedToFloat64Array();
test_fixedToFloat64Array_no_math_random();
test_sinhToFloat64Array();
test_sinhToFloat64Array_dither()
test_simpleRoundtripLossless();
console.log("All SerializationUtils tests passed");