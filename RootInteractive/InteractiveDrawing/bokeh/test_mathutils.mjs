import MathUtils from "./MathUtils.js"

function shallow_compare_absolute(A,B,delta){
	if(A === B) return true
	if(A.length !== B.length) return false
	return A.reduce((acc,cur,idx)=>(acc*idx+Math.abs(cur-B[idx]))/(idx+1), 0) <= delta
}

function test_chol(){
	let A = [1,0,1,0,0,1]
	let A_llt = [1,0,1,0,0,1]
	MathUtils.chol(A,3)
	console.assert(shallow_compare_absolute(A,A_llt,1e-6), "expected "+A+" got "+A_llt)	

}


test_chol()
