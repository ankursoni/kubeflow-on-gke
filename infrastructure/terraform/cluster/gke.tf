resource "google_container_cluster" "gke01" {
  name     = "${var.prefix}-${var.environment}-gke01"
  location = var.region

  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  node_locations           = [var.zone]

  network    = var.network_id
  subnetwork = var.subnet_id

  ip_allocation_policy {
    services_secondary_range_name = var.services_ip_range_name
    cluster_secondary_range_name  = var.pods_ip_range_name
  }

  release_channel {
    channel = "STABLE"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_global_access_config {
      enabled = false
    }
    master_ipv4_cidr_block = var.master_ip_range
  }

  lifecycle {
    ignore_changes = [
      initial_node_count
    ]
  }
}

resource "google_container_registry" "eugcr" {
  location = "EU"
}
resource "google_service_account" "nodepool_sa01" {
  account_id   = "nodepool-sa01"
  display_name = "Service Account for ${var.prefix}-${var.environment}-nodepool01, nodepool02 and nodepool03"
}
resource "google_storage_bucket_iam_member" "object_viewer01" {
  bucket = google_container_registry.eugcr.id
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.nodepool_sa01.email}"
}

resource "google_container_node_pool" "nodepool01" {
  name           = "${var.prefix}-${var.environment}-nodepool01"
  location       = var.region
  cluster        = google_container_cluster.gke01.name
  node_count     = var.node_count_pool01
  node_locations = [var.zone]

  node_config {
    machine_type = var.machine_type_pool01
    disk_type    = "pd-ssd"
    disk_size_gb = 50

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.nodepool_sa01.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    tags = ["${var.prefix}-${var.environment}-nodepool01"]

    labels = var.labels
  }

  autoscaling {
    min_node_count = var.node_count_pool01
    max_node_count = 5
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 1
  }

  lifecycle {
    ignore_changes = [
      node_count
    ]
  }
}

resource "google_container_node_pool" "nodepool02" {
  name           = "${var.prefix}-${var.environment}-nodepool02"
  location       = var.region
  cluster        = google_container_cluster.gke01.name
  node_count     = var.node_count_pool02
  node_locations = [var.zone]

  node_config {
    machine_type = var.machine_type_pool02
    disk_type    = "pd-ssd"
    disk_size_gb = 50

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.nodepool_sa01.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    tags = ["${var.prefix}-${var.environment}-nodepool02"]

    labels = var.labels
  }

  autoscaling {
    min_node_count = var.node_count_pool02
    max_node_count = 5
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 1
  }

  lifecycle {
    ignore_changes = [
      node_count
    ]
  }
}

resource "google_container_node_pool" "nodepool03" {
  name           = "${var.prefix}-${var.environment}-nodepool03"
  location       = var.region
  cluster        = google_container_cluster.gke01.name
  node_count     = var.node_count_pool03
  node_locations = [var.zone]

  node_config {
    machine_type = var.machine_type_pool03
    disk_type    = "pd-ssd"
    disk_size_gb = 50

    spot = true

    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.nodepool_sa01.email
    oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]

    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    tags = ["${var.prefix}-${var.environment}-nodepool02"]

    labels = var.labels
  }

  autoscaling {
    min_node_count = var.node_count_pool03
    max_node_count = 20
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }

  upgrade_settings {
    max_surge       = 1
    max_unavailable = 1
  }

  lifecycle {
    ignore_changes = [
      node_count
    ]
  }
}
resource "google_compute_firewall" "webhook_firewallrule" {
  name        = "${google_container_cluster.gke01.name}-webhook"
  network     = var.network_id
  description = "Port 8443 (kubeflow?), 15017 (istio), 4443 (seldon) for admission controller webhook"

  allow {
    protocol = "tcp"
    ports    = ["8443", "15017", "4443"]
  }

  source_ranges = [var.master_ip_range]
  target_tags = [
    google_container_node_pool.nodepool01.node_config[0].tags[0],
    google_container_node_pool.nodepool02.node_config[0].tags[0]
  ]
}

resource "google_service_account" "kfp_system_sa01" {
  account_id   = "gke01-kfp-system"
  display_name = "Service Account for kubeflow kfp-system"
}
resource "google_storage_bucket_iam_member" "object_viewer02" {
  bucket = var.storagebucket_id
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.kfp_system_sa01.email}"
}
resource "google_service_account_iam_binding" "kfp_system_iam01" {
  service_account_id = google_service_account.kfp_system_sa01.name
  role               = "roles/iam.workloadIdentityUser"
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[kubeflow/ml-pipeline-ui]",
    "serviceAccount:${var.project_id}.svc.id.goog[kubeflow/ml-pipeline-visualizationserver]",
  ]
}

resource "google_service_account" "kfp_user_sa01" {
  account_id   = "gke01-kfp-user"
  display_name = "Service Account for kubeflow kfp-user"
}
resource "google_storage_bucket_iam_member" "object_admin01" {
  bucket = var.storagebucket_id
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.kfp_user_sa01.email}"
}
resource "google_service_account_iam_binding" "kfp_user_iam01" {
  service_account_id = google_service_account.kfp_user_sa01.name
  role               = "roles/iam.workloadIdentityUser"
  members            = ["serviceAccount:${var.project_id}.svc.id.goog[kubeflow/pipeline-runner]"]
}
