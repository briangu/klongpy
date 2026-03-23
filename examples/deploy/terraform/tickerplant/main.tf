terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.28"
    }
  }
}

provider "kubernetes" {
  config_path = var.kubeconfig
}

variable "kubeconfig" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

resource "kubernetes_namespace" "klongpy" {
  metadata {
    name = "klongpy"
  }
}

resource "kubernetes_config_map" "tickerplant_script" {
  metadata {
    name      = "tickerplant-script"
    namespace = kubernetes_namespace.klongpy.metadata[0].name
  }

  data = {
    "srv_pubsub.kg" = file("../../../ipc/srv_pubsub.kg")
  }
}

resource "kubernetes_deployment" "tickerplant" {
  metadata {
    name      = "tickerplant"
    namespace = kubernetes_namespace.klongpy.metadata[0].name
    labels = {
      app = "tickerplant"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "tickerplant"
      }
    }

    template {
      metadata {
        labels = {
          app = "tickerplant"
        }
      }

      spec {
        container {
          name  = "tickerplant"
          image = "klongpy-app:latest"

          env {
            name  = "KG_FILE_PATH"
            value = "/scripts/srv_pubsub.kg"
          }

          port {
            container_port = 8888
          }

          volume_mount {
            name       = "script-volume"
            mount_path = "/scripts"
          }
        }

        volume {
          name = "script-volume"

          config_map {
            name = kubernetes_config_map.tickerplant_script.metadata[0].name
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "tickerplant" {
  metadata {
    name      = "tickerplant"
    namespace = kubernetes_namespace.klongpy.metadata[0].name
  }

  spec {
    selector = {
      app = "tickerplant"
    }

    port {
      port        = 8888
      target_port = 8888
    }

    type = "ClusterIP"
  }
}
