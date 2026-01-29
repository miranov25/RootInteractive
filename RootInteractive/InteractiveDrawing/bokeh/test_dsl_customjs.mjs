import fs from "fs"

const inputPath = process.argv[2];
const data = JSON.parse(fs.readFileSync(inputPath, "utf8"));

function shallow_compare_absolute(A,B,delta){
    if(A === B) return true
    if(A.length !== B.length) return false
    return A.reduce((acc,cur,idx)=>(acc*idx+Math.abs(cur-B[idx]))/(idx+1), 0) <= delta
}

for(const test_case of data.test_cases){
    const {A,B,delta,expected} = test_case
    const delta_val = typeof delta === "number" ? delta : 0.0
    const result = shallow_compare_absolute(A,B,delta_val)
    if(result !== expected){
        console.error(`Test case failed:
A: ${A}
B: ${B}
delta: ${delta}
expected: ${expected}
got: ${result}`)
        process.exit(1)
    } 
}

if(!data.test_cases.length){
    console.log("No test cases provided.")
}