package xiaozhi.modules.device.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.io.Serializable;

@Data
@Schema(description = "Change device agent request")
public class DeviceChangeAgentDTO implements Serializable {

    @NotBlank(message = "Device ID is required")
    @Schema(description = "Device ID")
    private String deviceId;

    @NotBlank(message = "New agent ID is required")
    @Schema(description = "New agent ID to migrate to")
    private String newAgentId;
}
