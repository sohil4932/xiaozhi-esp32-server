-- Hardware device registry table for pre-assigned device codes
-- Used for production devices that have codes printed on them during manufacturing
-- Also supports development flow via OTA random code generation (fallback)

CREATE TABLE IF NOT EXISTS `ai_device_hardware` (
    `id` VARCHAR(32) NOT NULL,
    `hardware_code` VARCHAR(32) NOT NULL COMMENT 'Pre-assigned code printed on device (e.g. 6-digit code)',
    `mac_address` VARCHAR(50) NOT NULL COMMENT 'Device MAC address',
    `firebase_uid` VARCHAR(128) DEFAULT NULL COMMENT 'Firebase user UID (set on bind)',
    `is_bound` TINYINT(1) NOT NULL DEFAULT 0 COMMENT '0=not bound, 1=bound to a user',
    `bound_at` DATETIME DEFAULT NULL COMMENT 'When the device was bound',
    `creator` BIGINT DEFAULT NULL,
    `create_date` DATETIME DEFAULT NULL,
    `updater` BIGINT DEFAULT NULL,
    `update_date` DATETIME DEFAULT NULL,
    PRIMARY KEY (`id`),
    UNIQUE INDEX `idx_hardware_code` (`hardware_code`),
    UNIQUE INDEX `idx_mac_address` (`mac_address`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Pre-registered hardware devices with printed codes';
