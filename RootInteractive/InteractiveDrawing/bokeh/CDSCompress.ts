import {ColumnarDataSource} from "models/sources/columnar_data_source"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"
//import * as pako from './pako/'
//declare var Buffer : any
declare const  pako : any
//const pako = require('pako');
//declare namespace pako {
//  function inflate(arrayOut:Uint8Array): Uint8Array;
//}

export namespace CDSCompress {
  export type Attrs = p.AttrsOf<Props>

  export type Props = ColumnarDataSource.Props & {
    source: p.Property<ColumnDataSource>
      inputData :    p.Property<Record<string, any>>
  }
}

export interface CDSCompress extends CDSCompress.Attrs {}

export class CDSCompress extends ColumnarDataSource {
  properties: CDSCompress.Props

  constructor(attrs?: Partial<CDSCompress.Attrs>) {
    super(attrs)
  }

  static __name__ = "CDSCompress"

  static init_CDSCompress() {

    this.define<CDSCompress.Props>(({Ref})=>({
      source:  [Ref(ColumnDataSource)],
        inputData:    [ p.Instance ]
    }))
  }

  initialize(): void {
    super.initialize()
    //this.data = {}
    console.info("CDSCompress::initialize")
    this.inflateCompressedBokehObjectBase64()
    //this.update_range()
  }
  inflateCompressedBokehBase64(arrayIn: any ) {
    let arrayOut=arrayIn.array
    for(var i =  arrayIn.actionArray.length-1;i>=0; i--) {
      console.log((i + 1) + " --> " + arrayIn.actionArray[i])
      if (arrayIn.actionArray[i][0] == "base64") {
        //arrayOut=Buffer.from(arrayOut, 'base64');
        arrayOut = atob(arrayOut).split("").map(function (x) {
          return x.charCodeAt(0);
        });
      }
      if (arrayIn.actionArray[i][0] == "zip") {
        //arrayOut=Buffer.from(arrayOut, 'base64');
        //var myArr = new Uint8Array(arrayOut)
        arrayOut = pako.inflate(arrayOut)
        if (arrayIn.indexType=="int16"){
          arrayOut=new Uint16Array(arrayOut.buffer);
        }
        if (arrayIn.indexType=="int32"){
          arrayOut=new Uint32Array(arrayOut.buffer);
        }
        console.log(arrayOut)
      }
      if (arrayIn.actionArray[i][0] == "code") {
        let size = arrayOut.length;
        let arrayOutNew = new Array();
        for (let i = 0; i < size; i++) {
          arrayOutNew.push(arrayIn.valueCode[arrayOut[i]]);
        }
        arrayOut=arrayOutNew
      }
    }
    return arrayOut
    //arrayIn.actionArray.reverse().forEach(x => console.log(x))
  }

  inflateCompressedBokehObjectBase64() {
    this.data = {};
    let objectOut0 = Object;
    let objectOut = (objectOut0 as any);
    let objectIn = (this.inputData as any);
    let length=0;
    for (let arr in objectIn) {
      let arrOut = this.inflateCompressedBokehBase64(objectIn[arr]);
      this.data[arr]=arrOut;
      //if (arrOut.length!=length)
      length=arrOut.length
      console.log(arr);
    }
    this.data["index"]=new Uint32Array(length)
     for (let i = 0; i < length; i++) {
         this.data["index"][i]=i;
        }
    return objectOut
  }

  public view: number[] | null

}
