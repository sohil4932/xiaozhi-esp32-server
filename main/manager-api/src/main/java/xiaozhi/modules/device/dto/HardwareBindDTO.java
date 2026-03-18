package xiaozhi.modules.device.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.io.Serializable;

@Data
@Schema(description = "Hardware device bind request")
public class HardwareBindDTO implements Serializable {

    @NotBlank(message = "Hardware code is required")
    @Schema(description = "Hardware code printed on device")
    private String hardwareCode;

    @Schema(description = "Firebase user UID")
    private String firebaseUid;

    @Schema(description = "Agent ID to bind to (optional, uses default agent if not provided)")
    private String agentId;
}
