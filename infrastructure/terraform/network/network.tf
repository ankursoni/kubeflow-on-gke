locals {
  master_ip_range = "172.16.0.0/28"
}

resource "google_compute_network" "network01" {
  name                    = "${var.prefix}-${var.environment}-vpc01"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "subnet01" {
  name                     = "${var.prefix}-${var.environment}-subnet01"
  ip_cidr_range            = "10.2.0.0/16"
  network                  = google_compute_network.network01.id
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "services-ip-range"
    ip_cidr_range = "192.168.0.0/22"
  }

  secondary_ip_range {
    range_name    = "pods-ip-range"
    ip_cidr_range = "192.168.64.0/18"
  }
}

resource "google_compute_router" "router01" {
  name    = "${var.prefix}-${var.environment}-router01"
  network = google_compute_network.network01.id
  region  = var.region
}

resource "google_compute_router_nat" "nat01" {
  name                               = "${var.prefix}-${var.environment}-nat01"
  router                             = google_compute_router.router01.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

resource "google_compute_address" "app_ip01" {
  name         = "${var.prefix}-${var.environment}-app-ip01"
  address_type = "EXTERNAL"
  network_tier = "PREMIUM"
}

output "network01_id" {
  value = google_compute_network.network01.id
}

output "subnet01_id" {
  value = google_compute_subnetwork.subnet01.id
}

output "master_ip_range" {
  value = local.master_ip_range
}

output "services_ip_range_name" {
  value = google_compute_subnetwork.subnet01.secondary_ip_range[0].range_name
}

output "pods_ip_range_name" {
  value = google_compute_subnetwork.subnet01.secondary_ip_range[1].range_name
}
