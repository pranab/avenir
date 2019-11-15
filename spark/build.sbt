name := "avenir-spark"

organization := "org.avenir"

version := "1.0"

scalaVersion := "2.12.0"

libraryDependencies ++=Seq(
  "org.apache.spark" %% "spark-core" % "3.0.0-preview" % "provided",
  "org.apache.commons" % "commons-lang3" % "3.0",
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.3.3",
  "com.fasterxml.jackson.module" % "jackson-module-scala_2.12" % "2.9.4",
  "junit" % "junit" % "4.7" % "test",
  "org.scalatest" % "scalatest_2.10" % "2.0" % "test",
  "org.chombo" %% "chombo-spark" % "1.0",
  "mawazo" %% "chombo" % "1.0",
  "mawazo" %% "avenir" % "1.0",
  "mawazo" %% "hoidla" % "1.0",
  "gov.nist.math" % "jama" % "1.0.3"
)
