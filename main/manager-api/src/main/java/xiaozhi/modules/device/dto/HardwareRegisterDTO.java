package xiaozhi.modules.device.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

import java.io.Serializable;

@Data
@Schema(description = "Hardware device pre-registration request (manufacturing)")
public class HardwareRegisterDTO implements Serializable {

    @NotBlank(message = "Hardware code is required")
    @Schema(description = "Hardware code to print on device")
    private String hardwareCode;

    @NotBlank(message = "MAC address is required")
    @Schema(description = "Device MAC address")
    private String macAddress;
}
