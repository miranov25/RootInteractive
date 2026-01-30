import fs from "fs"

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

function decodeColumn(column, data_source=null){
    if(typeof column === "object" && column.data && column.dtype === "float64"){
        return decodeBase64ToFloat64Array(column.data);
    }
    if(typeof column === "object" && column.func && column.args){
        const args_decoded = column.args.map(arg=>data_source.getColumn(arg));
        const func = new Function(...column.args, "$output", "data_source", column.func);
        return func(...args_decoded, null, data_source);
    }
    return column;
}

function DataFrame(data){
    this.data = data;
    this.getColumn = function(col_name){
        return decodeColumn(this.data[col_name], this);
    }
    this.get_length = function(){
        const first_key = Object.keys(this.data)[0];
        const col = this.getColumn(first_key);
        return col.length;
    }
}

for(const table_name in data.data){
    data.data[table_name] = new DataFrame(data.data[table_name]);
}

for(const test_case of data.test_cases){
    if(test_case.type === "EQ"){
        const table = data.data[test_case.table];
        const lhs_data = table.getColumn(test_case.lhs);
        const rhs_data = table.getColumn(test_case.rhs);
        const delta = test_case.epsilon || 1e-10;
        if(!shallow_compare_absolute(lhs_data, rhs_data, delta)){
            console.error(`Test case failed:
    lhs: ${test_case.lhs}
    rhs: ${test_case.rhs}
    delta: ${delta}`);
            process.exit(1);
        }
    } else {
        console.error(`Unknown test case type: ${test_case.type}`);
        process.exit(1);
    }
}

if(!data.test_cases.length){
    console.log("No test cases provided.")
}