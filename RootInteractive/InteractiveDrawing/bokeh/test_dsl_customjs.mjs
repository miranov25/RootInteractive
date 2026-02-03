import fs from "fs"
import zlib from "zlib"

const inputPath = process.argv[2];
const data = JSON.parse(fs.readFileSync(inputPath, "utf8"));

function shallow_compare_absolute(A,B,delta){
    if(A === B) return true
    if(!A || !B || A.length !== B.length) return false
    let m = 0;
    for (let i = 0; i < A.length; i++) {
        const d = Math.abs(A[i] - B[i]);
        if (d > m) m = d;
        if (m > delta) return false;
    }
    return true;
}

function decodeBase64ToFloat64Array(b64_string){
    const binary_string = Buffer.from(b64_string, 'base64');
    if (binary_string.length % 8 !== 0) {
        throw new Error(`Invalid float64 payload length ${binary_string.length} (not divisible by 8)`);
    }
    const float_array = new Float64Array(binary_string.buffer, binary_string.byteOffset, binary_string.length / Float64Array.BYTES_PER_ELEMENT);
    return float_array;
}

function decodeColumn(column, data_source=null, context={}){
    if(typeof column === "object" && column.data && column.dtype === "float64"){
        return decodeBase64ToFloat64Array(column.data);
    }
    if(typeof column === "object" && column.data && column.dtype === "int32"){
        const binary_string = Buffer.from(column.data, 'base64');  
        if (binary_string.length % 4 !== 0) {
            throw new Error(`Invalid int32 payload length ${binary_string.length} (not divisible by 4)`);
        }
        const int_array = new Int32Array(binary_string.buffer, binary_string.byteOffset, binary_string.length / Int32Array.BYTES_PER_ELEMENT);
        return int_array;
    }
    if(typeof column === "object" && column.func && column.args){
        const args_decoded = column.args.map(arg=>data_source.get_column(arg));
        const ctx_keys = Object.keys(column.context || {});
        const ctx_args = ctx_keys.map(key=>column.context[key]);
        const func = new Function(...column.args, ...ctx_keys, "$output", "data_source", column.func);
        return func(...args_decoded, ...ctx_args.map(path=>resolvePath(path, context)), null, data_source);
    }
    return column;
}

function resolvePath(path, context){
    const splits = path.split(".");
    let target = context;
    for(const split of splits){
        target = target[split];
    }
    return target;
}

function DataFrame(data, context={}){
    this.data = data;
    this.context = context;
    this.get_column = function(col_name){
        return decodeColumn(this.data[col_name], this, this.context);
    }
    this.get_length = function(){
        const first_key = Object.keys(this.data)[0];
        const col = this.get_column(first_key);
        return col.length;
    }
}

for(const table_name in data.data){
    data.data[table_name] = new DataFrame(data.data[table_name], data.data);
}

for(const test_case of data.test_cases){
    if(test_case.type === "EQ"){
        const table = data.data[test_case.table];
        const lhs_data = table.get_column(test_case.lhs);
        const rhs_data = table.get_column(test_case.rhs);
        const delta = test_case.epsilon || 1e-10;
        if(!shallow_compare_absolute(lhs_data, rhs_data, delta)){
            console.error(`Test case failed:
    lhs: ${test_case.lhs}
    rhs: ${test_case.rhs}
    delta: ${delta}`);
            process.exit(1);
        }
        console.log(`Test case passed:
    lhs: ${test_case.lhs}
    rhs: ${test_case.rhs}`);
    } else if(test_case.type === "ACTION"){
        const {key, new_value} = test_case;
        const key_splits = key.split(".");
        let target = data.data;
        for(let i=0; i<key_splits.length-1; i++){
            target = target[key_splits[i]];
        }
        target[key_splits[key_splits.length-1]] = new_value;
    } else {
        console.error(`Unknown test case type: ${test_case.type}`);
        process.exit(1);
    }
}

if(!data.test_cases.length){
    console.log("No test cases provided.")
}