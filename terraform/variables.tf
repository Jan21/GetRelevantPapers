variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "getrelevantpapers"
}

variable "container_port" {
  description = "Port exposed by the container"
  type        = number
  default     = 3444
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "min_capacity" {
  description = "Minimum number of instances"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum number of instances"
  type        = number
  default     = 3
}

variable "desired_capacity" {
  description = "Desired number of instances"
  type        = number
  default     = 1
}

variable "health_check_path" {
  description = "Health check endpoint"
  type        = string
  default     = "/"
}

variable "enable_bedrock" {
  description = "Enable AWS Bedrock integration"
  type        = bool
  default     = true
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the application"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production!
}

variable "domain_name" {
  description = "Optional domain name for the application"
  type        = string
  default     = ""
}

