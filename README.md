# Novelty Search - Evolution Strategy in Deep Reinforcement Learning

- Trong các bài toán Reinforcement Learning, mục tiêu của Agent là *cực đại hóa hàm phần thưởng*. Tuy nhiên, hàm phần thưởng có thể gây nhầm lẫn (bẫy) hoặc khan hiếm, khiến Agent kẹt ở điểm tối ưu cục bộ.

- Các giải pháp hiện đại thúc đẩy agent đến các trạng thái hiếm hoặc *chưa được khám phá* → khuyến khích khám phá (exploration)

- Một hướng tiếp cận khác là thiết kế hoặc học một mô tả về hành vi trong suốt lifetime của agent và thực hiện khám phá bằng một *quần thể các Agent*. Ví dụ: **Novelty Search (NS)** và **Quality Diversity (QD)**


<img src="https://i.imgur.com/9bEMmuf.png" alt="Minh hoạt NS" width="500"/>


---- 

# Các phương pháp tiếp cận

- NS-ES: Kết hợp Novelty Search và Evolution Strategy 
- NSR-ES: Lấy ý tưởng từ NS-ES nhưng bổ sung tín hiệu phần thưởng để cải thiện kết quả
- NS-ES customized: Kết hợp ý tưởng từ thuật giải di truyền và tiến hành huấn luyện theo từng quần thể (thay vì theo vòng lặp như NS-ES) để cải thiện kết quả của quần thể.
- NSR-ES customized: Lấy ý tưởng từ NS-ESc nhưng sử dụng tín hiệu phần thưởng để cải thiện kết quả

# Kết quả
- Thực hiện đánh giá trên 3 môi trường, bao gồm *MountainCar*, *CartPole* và *BreakoutNoFrameskip-v4* và so sánh với thuật toán DQN.

- Thuật toán NSR-ESc mặc dù kết quả thấp hơn DQN nhưng có tốc độ huấn luyện nhanh nhất và cho kết quả khá tốt, cho thấy tiềm năng để ứng dụng chiến lược *tìm kiếm tính mới* và *các thuật toán tiến hóa* để giải quyết các bài toán học tăng cường.

![Result](NSR-ESc.gif)