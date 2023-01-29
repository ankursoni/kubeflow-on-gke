variable "project_id" {
  default = ""
}
variable "region" {
  default = "europe-west4"
}
variable "zone" {
  default = "europe-west4-a"
}
variable "prefix" {
  default = "kubeflow"
}
variable "environment" {
  default = "prototype"
}
variable "machine_type_pool01" {
  default = "e2-medium"
}
variable "node_count_pool01" {
  type    = number
  default = 1
}
variable "machine_type_pool02" {
  default = "t2a-standard-1"
}
variable "node_count_pool02" {
  type    = number
  default = 1
}
variable "machine_type_pool03" {
  default = "t2a-standard-1"
}
variable "node_count_pool03" {
  type    = number
  default = 1
}
