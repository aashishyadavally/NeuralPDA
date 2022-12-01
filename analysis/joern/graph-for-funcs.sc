import gremlin.scala._
import org.apache.tinkerpop.gremlin.structure.Direction

import io.shiftleft.Implicits.JavaIteratorDeco
import io.shiftleft.codepropertygraph.generated._
import scala.io.Source
import scala.util.{Try,Success,Failure}
import java.nio.file.Paths

/* This file is APACHE licensed, based off of script from https://github.com/shiftleftsecurity/joern/ */

/** Some helper functions: adapted from ReachingDefPass.scala in codeproperty graph repo */
def vertexToStr(vertex: Vertex, identifiers: Map[Vertex,Int]): String = {
  val str = new StringBuffer()

  str.append(identifiers(vertex).toString)

  str.toString
}

def toProlog(graph: ScalaGraph): String = {
  var vertex_identifiers:Map[Vertex,Int] = Map()

  var index = 0
  graph.V.l.foreach{ v =>
    vertex_identifiers += (v -> index)
    index += 1
  }

  val buf = new StringBuffer()


  buf.append("% FEATURE\n")
  graph.V.l.foreach{ v =>
    // Try {
    //   buf.append(
    //      "\""
    //       +vertexToStr(v, vertex_identifiers)
    //       +"\":"
    //       + "[ \""
    //       + v.value2(NodeKeys.CODE).toString
    //       + "\","
    //       + " \""
    //       + v.value2(NodeKeys.TYPE_FULL_NAME).toString
    //       + "\","
    //       + " \""
    //       + v.value2(NodeKeys.NAME).toString
    //       + "\"]\n"
    //   )
    // }
    try {
        val code1 = v.value2(NodeKeys.CODE).toString.replace("\'", "")
        val code2 = code1.replace("\"", "")
        val code = code2.replace("\\", "")
        buf.append(
          "\""+
         vertexToStr(v, vertex_identifiers)
          +"\""
          +":"
          + "[ \""
          + code
          + "\","
      )
    }
    catch {
    case _: Throwable => buf.append(
          "\""+
         vertexToStr(v, vertex_identifiers)
          +"\""
          +":"
          + "[ \""
          + "0"
          + "\","
      )
    }
    try {
        buf.append(
          " \""
          + v.value2(NodeKeys.TYPE_FULL_NAME).toString
          + "\","
      )
    }
    catch {
    case _: Throwable => buf.append(
          " \""
          + "0"
          + "\","
      )
    }
    try {
        buf.append(
           " \""
           + v.value2(NodeKeys.NAME).toString
           + "\"]\n"
      )
    }
    catch {
    case _: Throwable => buf.append(
           " \""
           + "0"
           + "\"]\n"
      )
    }
  }

  buf.append("% AST\n")
  graph.E.hasLabel("AST").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

  buf.append("% CFG\n")
  graph.E.hasLabel("CFG").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

  buf.append("% REF\n")
  graph.E.hasLabel("REF").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

  buf.append("# CALL\n")
  graph.E.hasLabel("CALL").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }
//连接变量使用和定义位置，数据流分析的核心
  buf.append("# REACHING_DEF\n")
  graph.E.hasLabel("REACHING_DEF").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

//对应各种类型的节点到TYPE节点
  buf.append("# EVAL_TYPE\n")
  graph.E.hasLabel("EVAL_TYPE").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

//AST edges，比如block下面包含的所有statement都会有一条CONTAINS边
  buf.append("# CONTAINS\n")
  graph.E.hasLabel("CONTAINS").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

//这个边标志数据流的传播关系，人工定义，会在数据流分析中仔细分析
  buf.append("# PROPAGATE\n")
  graph.E.hasLabel("PROPAGATE").l.foreach { e =>
    val parentVertex = vertexToStr(e.outVertex, vertex_identifiers).replace("\"","\'")
    val childVertex = vertexToStr(e.inVertex, vertex_identifiers).replace("\"","\'")
    buf.append(s"""[$parentVertex, $childVertex]\n""")
  }

  buf.toString
}
@main def main(cpgFile: String): String = {
  loadCpg(cpgFile)
  toProlog(cpg.graph)
}