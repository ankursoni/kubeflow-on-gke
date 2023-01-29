locals {
  labels = {
    managed-by = "terraform"
  }
}

module "network" {
  source      = "./network"
  providers   = { google = google.default }
  labels      = local.labels
  region      = var.region
  zone        = var.zone
  prefix      = var.prefix
  environment = var.environment
}

module "storage" {
  source      = "./storage"
  providers   = { google = google.default }
  labels      = local.labels
  region      = var.region
  zone        = var.zone
  prefix      = var.prefix
  environment = var.environment
}

module "cluster" {
  source                 = "./cluster"
  providers              = { google = google.default }
  labels                 = local.labels
  project_id             = var.project_id
  region                 = var.region
  zone                   = var.zone
  prefix                 = var.prefix
  environment            = var.environment
  network_id             = module.network.network01_id
  subnet_id              = module.network.subnet01_id
  master_ip_range        = module.network.master_ip_range
  services_ip_range_name = module.network.services_ip_range_name
  pods_ip_range_name     = module.network.pods_ip_range_name
  machine_type_pool01    = var.machine_type_pool01
  node_count_pool01      = var.node_count_pool01
  machine_type_pool02    = var.machine_type_pool02
  node_count_pool02      = var.node_count_pool02
  machine_type_pool03    = var.machine_type_pool03
  node_count_pool03      = var.node_count_pool03
  storagebucket_id       = module.storage.storagebucket01_id

  depends_on = [module.network, module.storage]
}
