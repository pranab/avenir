organization := "mawazo"

name := "avenir"

version := "1.0"

scalaVersion := "2.10.4"

sources in (Compile, doc) ~= (_ filter (_.getName endsWith ".scala"))

packageBin in Compile := file(s"target/${name.value}-${version.value}.jar")