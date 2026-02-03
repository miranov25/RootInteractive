import { pathToFileURL } from "url";
import path from "path";

const tempdir = process.argv[2];

const file = path.join(tempdir, "MathUtils.js");

const MathUtils = await import(pathToFileURL(file).href);

function shallow_compare_absolute(A,B,delta){
	if(A === B) return true
	if(A.length !== B.length) return false
	return A.reduce((acc,cur,idx)=>(acc*idx+Math.abs(cur-B[idx]))/(idx+1), 0) <= delta
}

function assert(cond, msg) {
  if (!cond) {
    throw new Error(msg);
  }
}

function test_chol(){
	// Identity matrix
	let A = [1,0,1,0,0,1]
	let A_llt = [1,0,1,0,0,1]
	MathUtils.chol(A,3)
	assert(shallow_compare_absolute(A,A_llt,1e-6), "expected "+A_llt+" got "+A)	
}

function test_unbinned_quantile(){
	let A = [1,2,3,4,5,6,7,8,9]

	let q = MathUtils.quantile(A,0.5)
	assert(q === 5, "expected 5 got "+q)

	q = MathUtils.quantile(A, 0.25, 0, -1)
	assert(q === 3, "expected 3 got "+q)

	q = MathUtils.quantile(A,0.75, 0, -1)
	assert(q === 7, "expected 7 got "+q)

	q = MathUtils.quantile(A,0, 0, -1)
	assert(q === 1, "expected 1 got "+q)

	q = MathUtils.quantile(A,1, 0, -1)
	assert(q === 9, "expected 9 got "+q)

	q = MathUtils.quantile(A,0.0625, 0, -1)
	assert(q === 1.5, "expected 1.5 got "+q)
}

test_chol()
test_unbinned_quantile()
console.log("All MathUtils tests passed")
