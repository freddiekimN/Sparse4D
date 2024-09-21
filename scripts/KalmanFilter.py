import numpy as np

class UnscentedKalmanFilter:
    def __init__(self, dt,initial_state, process_noise_std, measurement_noise_std):
        self.dt = dt
        self.n = 6  # 상태 벡터 차원 [x, y, v, acc, yaw, yaw rate]
        self.kappa = 0  # Scaling parameter
        
        # 초기 상태 벡터 [x, y, v, acc, yaw, yaw rate]
        self.x = np.array(initial_state)

        # 초기 공분산 행렬
        self.P = np.eye(self.n) * 1000

        # 프로세스 노이즈 공분산 행렬
        self.Q = np.eye(self.n) * process_noise_std
        self.Q[4, 4] *= 5  # yaw의 예측 오차를 키워서 측정값을 더 신뢰하게 함

        # 측정 노이즈 공분산 행렬
        self.R = np.diag(measurement_noise_std)
        self.R[1, 1] *= 0.5  # yaw rate 측정 노이즈를 줄여서 필터가 측정값에 더 의존하도록 함

        # 측정 행렬 H (속도와 yaw rate만 측정)
        self.H = np.array([
            [0, 0, 1, 0, 0, 0],  # 속도 v
            [0, 0, 0, 0, 0, 1]   # yaw rate
        ])

    def sigma_points(self, x, P):
        """ 시그마 포인트 생성 """
        lambda_ = self.kappa - self.n
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = x
        
        sqrt_P = np.linalg.cholesky((self.n + lambda_) * P)
        for i in range(self.n):
            sigma_points[i + 1] = x + sqrt_P[i]
            sigma_points[self.n + i + 1] = x - sqrt_P[i]
        
        return sigma_points

    def predict_sigma_points(self, sigma_points, u):
        """ 시그마 포인트 예측 단계 """
        sigma_points_pred = np.zeros_like(sigma_points)
        for i, point in enumerate(sigma_points):
            x, y, v, acc, yaw, yaw_rate = point
            
            # 상태 전이 방정식 적용
            v_pred = v + acc * self.dt
            yaw_pred = yaw + yaw_rate * self.dt
            x_pred = x + v_pred * np.cos(yaw) * self.dt
            y_pred = y + v_pred * np.sin(yaw) * self.dt
            yaw_rate_pred = yaw_rate + u[1]  # yaw rate 제어 입력을 반영
            
            sigma_points_pred[i] = [x_pred, y_pred, v_pred, acc, yaw_pred, yaw_rate_pred]
        
        return sigma_points_pred

    def predict(self, u):
        """ 예측 단계 """
        lambda_ = self.kappa - self.n
        sigma_points = self.sigma_points(self.x, self.P)
        sigma_points_pred = self.predict_sigma_points(sigma_points, u)
        
        # 가중치 계산
        Wm = np.full(2 * self.n + 1, 0.5 / (self.n + lambda_))
        Wm[0] = lambda_ / (self.n + lambda_)
        Wc = np.copy(Wm)
        
        # 예측 상태와 공분산 계산
        self.x = np.sum(Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        self.P = self.Q + sum(Wc[i] * np.outer(sigma_points_pred[i] - self.x, sigma_points_pred[i] - self.x) 
                             for i in range(2 * self.n + 1))

    def update(self, z):
        """ 업데이트 단계 """
        lambda_ = self.kappa - self.n
        sigma_points = self.sigma_points(self.x, self.P)
        sigma_points_pred = self.predict_sigma_points(sigma_points, [0, 0])

        # 측정 예측
        z_pred = np.dot(self.H, sigma_points_pred.T).dot([0.5 / (self.n + lambda_)] * (2 * self.n + 1))

        # 측정 공분산
        P_zz = self.R + sum(0.5 / (self.n + lambda_) * np.outer(np.dot(self.H, sigma_points_pred[i]) - z_pred,
                                                                np.dot(self.H, sigma_points_pred[i]) - z_pred)
                            for i in range(2 * self.n + 1))

        # 상태-측정 공분산
        P_xz = sum(0.5 / (self.n + lambda_) * np.outer(sigma_points_pred[i] - self.x,
                                                       np.dot(self.H, sigma_points_pred[i]) - z_pred)
                   for i in range(2 * self.n + 1))

        # 칼만 게인
        K = P_xz @ np.linalg.inv(P_zz)

        # 상태 업데이트
        self.x += K @ (z - z_pred)

        # 공분산 업데이트
        self.P -= K @ P_zz @ K.T

    def get_state(self):
        """ 현재 상태를 반환합니다. """
        return self.x
    
class LinearKalmanFilter:
    def __init__(self, dt,initial_state, process_noise_std, measurement_noise_std):
        self.dt = dt
        self.n = 6  # 상태 벡터 차원 [x, y, v, acc, yaw, yaw rate]
        
        # 초기 상태 벡터 [x, y, v, acc, yaw, yaw rate]
        self.x = np.array(initial_state)

        # 초기 공분산 행렬
        self.P = np.eye(self.n) * 1000

        # 프로세스 노이즈 공분산 행렬
        self.Q = np.eye(self.n) * process_noise_std

        # 측정 노이즈 공분산 행렬
        self.R = np.diag(measurement_noise_std)

        # 측정 행렬 H (속도와 yaw rate만 측정)
        self.H = np.array([
            [0, 0, 1, 0, 0, 0],  # 속도 v
            [0, 0, 0, 0, 0, 1]   # yaw rate
        ])

        self.A = np.array([
            [1, 0, self.dt * np.cos(np.radians(self.x[4])), 0, 0, 0],  # x
            [0, 1, self.dt * np.sin(np.radians(self.x[4])), 0, 0, 0],  # y
            [0, 0, 1, self.dt, 0, 0],                     # v
            [0, 0, 0, 1, 0, 0],                           # acc
            [0, 0, 0, 0, 1, self.dt],                     # yaw
            [0, 0, 0, 0, 0, 1]                            # yaw rate
        ])    
        # 제어 입력 행렬 B
        self.B = np.array([
            [0, 0],
            [0, 0],
            [1, 0],  # v 제어
            [0, 0],
            [0, 0],
            [0, 1]   # yaw rate 제어
        ])        

    def set_init(self, initial_state):
        self.x = np.array(initial_state)

    def predict(self,dt,u):
        """ 예측 단계 """
        
        self.dt = dt
        # 상태 전이 행렬 A
        self.A = np.array([
            [1, 0, self.dt * np.cos(np.radians(self.x[4])), 0, 0, 0],  # x
            [0, 1, self.dt * np.sin(np.radians(self.x[4])), 0, 0, 0],  # y
            [0, 0, 1, self.dt, 0, 0],                     # v
            [0, 0, 0, 1, 0, 0],                           # acc
            [0, 0, 0, 0, 1, self.dt],                     # yaw
            [0, 0, 0, 0, 0, 1]                            # yaw rate
        ])        
        
        # xx = np.dot(self.B, u)
        # self.x[5] = xx[5]
        # 상태 예측
        self.x = np.dot(self.A, self.x)

        # 공분산 예측
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def update(self, z):
        """ 업데이트 단계 """
        # 칼만 게인 계산
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 상태 업데이트
        y = z - np.dot(self.H, self.x)  # 잔차 (측정값과 예측값의 차이)
        self.x += np.dot(K, y)

        # 공분산 업데이트
        I = np.eye(self.n)
        self.P = np.dot(I - np.dot(K, self.H), self.P)

    def get_state(self):
        """ 현재 상태를 반환합니다. """
        return self.x    