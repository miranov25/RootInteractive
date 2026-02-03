import { pathToFileURL } from "url";
import path from "path";

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

test_fixedToFloat64Array();
console.log("All SerializationUtils tests passed");