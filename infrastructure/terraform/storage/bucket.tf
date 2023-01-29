resource "google_storage_bucket" "storagebucket01" {
  name                        = "${var.prefix}-${var.environment}-storagebucket01"
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = true

  labels = var.labels
}

output "storagebucket01_id" {
  value = google_storage_bucket.storagebucket01.id
}
