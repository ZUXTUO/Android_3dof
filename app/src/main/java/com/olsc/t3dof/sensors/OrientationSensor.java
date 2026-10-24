package com.olsc.t3dof.sensors;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.opengl.Matrix;
import android.view.Surface;

/**
 * 负责处理设备传感器数据以计算设备的 3D 姿态（3DOF）。
 *
 * 为了兼容不同硬件（尤其是没有磁力计 / 旋转矢量的设备，如 Rokid Glass 3），
 * 采用多级回退策略，按优先级自动选择可用的姿态来源：
 *
 *   1. 旋转矢量（TYPE_ROTATION_VECTOR）        —— 绝对姿态，精度最高，无漂移
 *   2. 游戏旋转矢量（TYPE_GAME_ROTATION_VECTOR）—— 陀螺仪 + 加速度计融合，相对姿态，yaw 会缓慢漂移
 *   3. 加速度计 + 磁力计                        —— 需要磁力计
 *   4. 仅陀螺仪积分                             —— 整体会随时间漂移
 *   5. 仅重力矢量（加速度计 / 重力传感器）      —— 最后兜底，只有俯仰+滚转（2DOF），无航向
 *
 * 缺少磁力计时无法获得绝对航向，但不影响基础的 3DOF 旋转跟随效果。
 * 若连陀螺仪都没有（如低价 Android 手表），则退化为第 5 级：用重力方向解算
 * 倾斜姿态，画面仍会跟随设备的俯仰/滚转，但绕竖直轴转动时画面不跟随。
 */
public class OrientationSensor implements SensorEventListener {

    /** 姿态来源模式。 */
    public enum Mode {
        /** 无任何可用传感器，场景保持静止。 */
        NONE,
        /** 旋转矢量（含磁力计，绝对航向）。 */
        ROTATION_VECTOR,
        /** 游戏旋转矢量（陀螺仪 + 加速度计，相对姿态）。 */
        GAME_ROTATION_VECTOR,
        /** 加速度计 + 磁力计。 */
        ACCEL_MAG,
        /** 仅陀螺仪积分。 */
        GYROSCOPE,
        /** 仅重力矢量（加速度计或重力传感器），只有俯仰/滚转。 */
        ACCEL_ONLY
    }

    private SensorManager sensorManager;
    private Sensor rotationSensor;
    private Sensor gameRotationSensor;
    private Sensor accelerometer;
    private Sensor magnetometer;
    private Sensor gyroscope;
    private Sensor gravitySensor;

    private Mode mode = Mode.NONE;

    private final float[] rotationMatrix = new float[16];
    private final float[] orientation = new float[3];

    private final float[] gravity = new float[3];
    private final float[] geomagnetic = new float[3];
    private boolean hasGravity = false;
    private boolean hasGeomagnetic = false;

    // 仅重力模式相关：低通滤波后的重力矢量
    private final float[] filteredGravity = new float[3];
    private boolean hasFilteredGravity = false;

    // 陀螺仪积分相关
    private long lastGyroTimestamp = 0L;
    private final float[] deltaMatrix = new float[16];
    private final float[] tempMatrix = new float[16];

    /** 传感器是否已生成有效的姿态数据。 */
    private boolean ready = false;

    public OrientationSensor() {
        // 初始化旋转矩阵为单位矩阵
        Matrix.setIdentityM(rotationMatrix, 0);
    }

    /**
     * 检测设备可用的姿态来源，并按优先级选择模式（不会注册监听器）。
     *
     * @return 选中的模式
     */
    public Mode detectMode(Context context) {
        SensorManager sm = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        if (sm == null) {
            mode = Mode.NONE;
            return mode;
        }

        rotationSensor = sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        gameRotationSensor = sm.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR);
        accelerometer = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        magnetometer = sm.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        gyroscope = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        gravitySensor = sm.getDefaultSensor(Sensor.TYPE_GRAVITY);

        logAvailableSensors(sm);

        if (rotationSensor != null) {
            mode = Mode.ROTATION_VECTOR;
        } else if (gameRotationSensor != null) {
            mode = Mode.GAME_ROTATION_VECTOR;
        } else if (accelerometer != null && magnetometer != null) {
            mode = Mode.ACCEL_MAG;
        } else if (gyroscope != null) {
            mode = Mode.GYROSCOPE;
        } else if (accelerometer != null || gravitySensor != null) {
            // 只有重力方向可用：能解算俯仰/滚转，无法解算航向
            mode = Mode.ACCEL_ONLY;
        } else {
            mode = Mode.NONE;
        }
        return mode;
    }

    /**
     * 打印设备上报的全部传感器，便于排查“系统设置说有、应用却检测不到”的差异。
     */
    private void logAvailableSensors(SensorManager sm) {
        java.util.List<Sensor> all = sm.getSensorList(Sensor.TYPE_ALL);
        StringBuilder sb = new StringBuilder("设备传感器共 ").append(all.size()).append(" 个: ");
        for (Sensor s : all) {
            sb.append('[').append(s.getType()).append(' ').append(s.getName()).append("] ");
        }
        android.util.Log.d("OrientationSensor", sb.toString());
    }

    /** 返回当前（或最近一次检测到的）模式。 */
    public Mode getMode() {
        return mode;
    }

    /** 模式的易读名称，用于提示。 */
    public String getModeName() {
        switch (mode) {
            case ROTATION_VECTOR:
                return "旋转矢量（绝对姿态）";
            case GAME_ROTATION_VECTOR:
                return "游戏旋转矢量（相对姿态）";
            case ACCEL_MAG:
                return "加速度计 + 磁力计";
            case GYROSCOPE:
                return "陀螺仪积分";
            case ACCEL_ONLY:
                return "仅重力矢量（俯仰/滚转）";
            default:
                return "无可用传感器";
        }
    }

    /**
     * 当前模式的局限性提示。若该模式没有需要提醒的问题，返回 null。
     */
    public String getModeNotice() {
        switch (mode) {
            case GAME_ROTATION_VECTOR:
                return "未检测到磁力计或旋转矢量，已使用游戏旋转矢量：航向 (yaw) 无绝对参考且会缓慢漂移。";
            case GYROSCOPE:
                return "未检测到旋转矢量与磁力计，已退化为陀螺仪积分：姿态会随时间漂移。";
            case ACCEL_ONLY:
                return "未检测到陀螺仪，仅用重力方向解算倾斜：跟随俯仰/滚转，绕竖直轴转动（航向）时不跟随。";
            case ROTATION_VECTOR:
            case ACCEL_MAG:
                return null;
            default:
                return "未检测到任何可用于 3DOF 的姿态传感器。";
        }
    }

    /**
     * 注册传感器监听器并开始获取数据。
     */
    public void start(Context context) {
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        if (sensorManager == null) {
            mode = Mode.NONE;
            return;
        }

        // 每次启动重新探测，保证传感器引用有效
        detectMode(context);

        switch (mode) {
            case ROTATION_VECTOR:
                sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_GAME);
                break;
            case GAME_ROTATION_VECTOR:
                sensorManager.registerListener(this, gameRotationSensor, SensorManager.SENSOR_DELAY_GAME);
                break;
            case ACCEL_MAG:
                if (accelerometer != null) {
                    sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
                }
                if (magnetometer != null) {
                    sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_GAME);
                }
                break;
            case GYROSCOPE:
                // 陀螺仪积分从当前姿态开始，重置时间戳
                lastGyroTimestamp = 0L;
                Matrix.setIdentityM(rotationMatrix, 0);
                sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_GAME);
                break;
            case ACCEL_ONLY:
                // 优先用系统已滤好的重力传感器，退化时对原始加速度做低通滤波
                hasFilteredGravity = false;
                if (gravitySensor != null) {
                    sensorManager.registerListener(this, gravitySensor, SensorManager.SENSOR_DELAY_GAME);
                } else if (accelerometer != null) {
                    sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_GAME);
                }
                break;
            default:
                android.util.Log.w("OrientationSensor", "无可用姿态传感器，3DOF 将保持静止");
                break;
        }
    }

    /**
     * 注销监听器并释放资源。
     */
    public void stop() {
        if (sensorManager != null) {
            sensorManager.unregisterListener(this);
        }
        lastGyroTimestamp = 0L;
    }

    /**
     * 检查传感器是否已经产生了第一个有效读数。
     */
    public boolean isReady() {
        return ready;
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        switch (event.sensor.getType()) {
            case Sensor.TYPE_ROTATION_VECTOR:
            case Sensor.TYPE_GAME_ROTATION_VECTOR:
                // 两者都直接映射为旋转矩阵
                SensorManager.getRotationMatrixFromVector(rotationMatrix, event.values);
                markReady(event.sensor.getType());
                break;

            case Sensor.TYPE_ACCELEROMETER:
                if (mode == Mode.ACCEL_ONLY) {
                    // 单加速度计模式：先滤出重力方向，再解算倾斜
                    lowPassGravity(event.values);
                    if (updateRotationMatrixFromGravity(filteredGravity)) {
                        markReady(event.sensor.getType());
                    }
                    break;
                }
                System.arraycopy(event.values, 0, gravity, 0, 3);
                hasGravity = true;
                updateRotationMatrixFromAccelMag();
                break;

            case Sensor.TYPE_GRAVITY:
                // 系统已融合好的重力矢量，直接解算倾斜
                if (updateRotationMatrixFromGravity(event.values)) {
                    markReady(event.sensor.getType());
                }
                break;

            case Sensor.TYPE_MAGNETIC_FIELD:
                System.arraycopy(event.values, 0, geomagnetic, 0, 3);
                hasGeomagnetic = true;
                updateRotationMatrixFromAccelMag();
                break;

            case Sensor.TYPE_GYROSCOPE:
                integrateGyroscope(event);
                break;

            default:
                break;
        }
    }

    private void markReady(int sensorType) {
        if (!ready) {
            ready = true;
            android.util.Log.d("OrientationSensor", "传感器就绪（" + getModeName() + "，type=" + sensorType + "）");
        }
    }

    /**
     * 当同时拥有重力和地磁数据时，手动更新旋转矩阵。
     */
    private void updateRotationMatrixFromAccelMag() {
        if (hasGravity && hasGeomagnetic) {
            float[] R = new float[16];
            float[] I = new float[16];
            if (SensorManager.getRotationMatrix(R, I, gravity, geomagnetic)) {
                System.arraycopy(R, 0, rotationMatrix, 0, 16);
                markReady(Sensor.TYPE_ACCELEROMETER);
            }
        }
    }

    /**
     * 一阶低通滤波，从原始加速度中提取重力方向，抑制手部运动产生的线性加速度。
     */
    private void lowPassGravity(float[] a) {
        final float alpha = 0.85f;
        if (!hasFilteredGravity) {
            System.arraycopy(a, 0, filteredGravity, 0, 3);
            hasFilteredGravity = true;
            return;
        }
        for (int i = 0; i < 3; i++) {
            filteredGravity[i] = alpha * filteredGravity[i] + (1f - alpha) * a[i];
        }
    }

    /**
     * 仅用重力矢量解算姿态（俯仰 + 滚转，航向恒为 0）。
     *
     * 参考姿态为“设备平放、屏幕朝上”（单位阵）。做法是求把设备坐标系下的天顶
     * 单位向量 n 旋转到世界 +Z 轴的<b>最小旋转</b>：绕水平轴 n×(0,0,1) 旋转
     * acos(n·Z)。该旋转不引入绕世界 Z 轴的扭转，因此等价于“pitch/roll 来自
     * 重力，yaw 固定为 0”，且天然正交、无需再正交化。
     *
     * @param g 设备坐标系下的加速度/重力矢量（静止时指向天空）
     * @return 是否成功更新
     */
    private boolean updateRotationMatrixFromGravity(float[] g) {
        float gx = g[0], gy = g[1], gz = g[2];
        float len = (float) Math.sqrt(gx * gx + gy * gy + gz * gz);
        if (len < 1.0f) {
            // 接近自由落体，方向不可信
            return false;
        }
        float nx = gx / len, ny = gy / len, nz = gz / len;

        Matrix.setIdentityM(rotationMatrix, 0);

        if (nz > 0.99999f) {
            // 近似平放：单位阵
            return true;
        }
        if (nz < -0.99999f) {
            // 屏幕朝下：绕 X 轴翻转 180°，避免退化
            rotationMatrix[5] = -1f;
            rotationMatrix[10] = -1f;
            return true;
        }

        // nz→-1 时分母趋零，上面已排除
        float inv = 1f / (1f + nz);
        rotationMatrix[0] = nz + ny * ny * inv;
        rotationMatrix[1] = -nx * ny * inv;
        rotationMatrix[2] = -nx;
        rotationMatrix[4] = -nx * ny * inv;
        rotationMatrix[5] = nz + nx * nx * inv;
        rotationMatrix[6] = -ny;
        rotationMatrix[8] = nx;
        rotationMatrix[9] = ny;
        rotationMatrix[10] = nz;
        return true;
    }

    /**
     * 陀螺仪积分：以机体坐标系下的角速度增量为单位四元数，右乘到当前姿态上。
     */
    private void integrateGyroscope(SensorEvent event) {
        long now = event.timestamp;
        if (lastGyroTimestamp == 0L) {
            lastGyroTimestamp = now;
            return;
        }
        float dt = (now - lastGyroTimestamp) * 1.0e-9f;
        lastGyroTimestamp = now;
        // 过滤异常时间间隔
        if (dt <= 0f || dt > 0.5f) {
            return;
        }

        // 小角度近似：旋转矢量 ≈ 角速度 × dt
        float[] deltaVector = {
                event.values[0] * dt,
                event.values[1] * dt,
                event.values[2] * dt
        };
        SensorManager.getRotationMatrixFromVector(deltaMatrix, deltaVector);

        // R = R * deltaR（机体坐标系增量）
        Matrix.multiplyMM(tempMatrix, 0, rotationMatrix, 0, deltaMatrix, 0);
        System.arraycopy(tempMatrix, 0, rotationMatrix, 0, 16);

        // 定期正交化，抑制数值漂移导致的形变
        orthonormalize(rotationMatrix);

        markReady(Sensor.TYPE_GYROSCOPE);
    }

    /**
     * 对旋转矩阵的 3x3 部分做 Gram-Schmidt 正交化，避免积分误差累积导致形变。
     */
    private void orthonormalize(float[] m) {
        float c0x = m[0], c0y = m[1], c0z = m[2];
        float c1x = m[4], c1y = m[5], c1z = m[6];

        // 归一化第一列
        float len0 = (float) Math.sqrt(c0x * c0x + c0y * c0y + c0z * c0z);
        if (len0 < 1.0e-6f) {
            return;
        }
        c0x /= len0; c0y /= len0; c0z /= len0;

        // 第二列减去在第一列上的投影并归一化
        float d = c1x * c0x + c1y * c0y + c1z * c0z;
        c1x -= d * c0x;
        c1y -= d * c0y;
        c1z -= d * c0z;
        float len1 = (float) Math.sqrt(c1x * c1x + c1y * c1y + c1z * c1z);
        if (len1 < 1.0e-6f) {
            return;
        }
        c1x /= len1; c1y /= len1; c1z /= len1;

        // 第三列 = 第一列 × 第二列
        float c2x = c0y * c1z - c0z * c1y;
        float c2y = c0z * c1x - c0x * c1z;
        float c2z = c0x * c1y - c0y * c1x;

        m[0] = c0x; m[1] = c0y; m[2] = c0z;
        m[4] = c1x; m[5] = c1y; m[6] = c1z;
        m[8] = c2x; m[9] = c2y; m[10] = c2z;
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // 暂时无需处理精度变化
    }

    /**
     * 获取当前的旋转矩阵。
     */
    public float[] getRotationMatrix() {
        return rotationMatrix;
    }

    /**
     * 获取当前设备的欧拉角 [方位角, 俯仰角, 滚转角]。
     */
    public float[] getOrientation() {
        SensorManager.getOrientation(rotationMatrix, orientation);
        return orientation;
    }

    /**
     * 根据当前屏幕旋转角度（横屏/竖屏）调整传感器旋转矩阵。
     *
     * @param rotation Surface.ROTATION_0/90/180/270
     */
    public float[] getAdjustedRotationMatrix(int rotation) {
        float[] adjustedMatrix = new float[16];
        System.arraycopy(rotationMatrix, 0, adjustedMatrix, 0, 16);

        switch (rotation) {
            case Surface.ROTATION_0:
                break;
            case Surface.ROTATION_90:
                SensorManager.remapCoordinateSystem(rotationMatrix, SensorManager.AXIS_Y, SensorManager.AXIS_MINUS_X,
                        adjustedMatrix);
                break;
            case Surface.ROTATION_180:
                SensorManager.remapCoordinateSystem(rotationMatrix, SensorManager.AXIS_MINUS_X,
                        SensorManager.AXIS_MINUS_Y, adjustedMatrix);
                break;
            case Surface.ROTATION_270:
                SensorManager.remapCoordinateSystem(rotationMatrix, SensorManager.AXIS_MINUS_Y, SensorManager.AXIS_X,
                        adjustedMatrix);
                break;
            default:
                break;
        }

        return adjustedMatrix;
    }

    /**
     * 检查设备是否具备运行 3DOF 所需的传感器。
     *
     * 只要满足以下任意一种即可：
     *   旋转矢量 / 游戏旋转矢量 / (加速度计 + 磁力计) / 陀螺仪
     */
    public boolean hasRequiredSensors(Context context) {
        return detectMode(context) != Mode.NONE;
    }
}
