provider "google" {
  alias   = "default"
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
