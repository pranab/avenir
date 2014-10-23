import sbtassembly.{Plugin},Plugin.AssemblyKeys._
import sbtassembly.Plugin._

name := "avenir-spark"

organization := "org.avenir"

version := "1.0"

scalaVersion := "2.10.4"

libraryDependencies ++=Seq(
  "org.apache.spark" %% "spark-core" % "1.1.0" % "provided",
  "org.apache.spark" %% "spark-streaming" % "1.1.0",
  "org.apache.spark" %% "spark-streaming-kafka" % "1.1.0",
  "org.apache.commons" % "commons-lang3" % "3.0",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.3.3",
  "com.fasterxml.jackson.module" % "jackson-module-scala_2.10" % "2.3.3",
  "junit" % "junit" % "4.7" % "test",
  "org.scalatest" % "scalatest_2.10" % "2.0" % "test"
)

net.virtualvoid.sbt.graph.Plugin.graphSettings

assemblySettings
