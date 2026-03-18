package xiaozhi.modules.device.entity;

import java.util.Date;

import com.baomidou.mybatisplus.annotation.FieldFill;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.annotation.TableField;
import com.baomidou.mybatisplus.annotation.TableId;
import com.baomidou.mybatisplus.annotation.TableName;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("ai_device_hardware")
@Schema(description = "Pre-registered hardware device")
public class DeviceHardwareEntity {

    @TableId(type = IdType.ASSIGN_UUID)
    @Schema(description = "ID")
    private String id;

    @Schema(description = "Hardware code printed on device")
    private String hardwareCode;

    @Schema(description = "MAC address")
    private String macAddress;

    @Schema(description = "Firebase user UID")
    private String firebaseUid;

    @Schema(description = "Bound status (0=unbound, 1=bound)")
    private Integer isBound;

    @Schema(description = "Bind timestamp")
    private Date boundAt;

    @Schema(description = "Creator")
    @TableField(fill = FieldFill.INSERT)
    private Long creator;

    @Schema(description = "Create date")
    @TableField(fill = FieldFill.INSERT)
    private Date createDate;

    @Schema(description = "Updater")
    @TableField(fill = FieldFill.UPDATE)
    private Long updater;

    @Schema(description = "Update date")
    @TableField(fill = FieldFill.UPDATE)
    private Date updateDate;
}
