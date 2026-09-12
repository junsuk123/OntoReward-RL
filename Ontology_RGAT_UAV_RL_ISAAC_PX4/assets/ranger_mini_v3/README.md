# AGILEX RANGER MINI 3.0 model

[기본 프로젝트 README](../../README.md) ·
[S5 기본 route 감사 그림](../../docs/images/metasejong_gwanggaeto_ugv_route.png)

`source/` 아래 파일은 AGILEX Robotics 공개
[`ugv_gazebo_sim`](https://github.com/agilexrobotics/ugv_gazebo_sim) 저장소의
commit `27633a956c845903ee630538afeb17fe70afdd84`에서 복사한
`ranger_mini_v3` package다. Upstream `package.xml`은 license를 BSD로 선언한다.
파일은 변경 없이 보존하며 `ranger_base.zip`은 upstream `ranger_base.dae` archive다.

고정된 source로 Isaac Sim USD를 생성한다.

```bash
./scripts/import_ranger_mini_v3.sh
```

Simulator는 `ranger_mini_v3.usd`를 kinematic landing platform의 visual-only
child로 참조한다. 생성된 asset이 없으면 동일한 외부 치수와 URDF wheel 위치를
가진 감사된 primitive로 대체하므로 visual asset 누락으로 실험이 중단되지 않는다.

기본 `config/shin2026-system.yaml` profile에서 이 visual은 Meta-Sejong
S5/Gwanggaeto 37-point 도로 폐곡선을 달리며 1.5×1.5 m landing deck를 싣는다.
Trajectory dynamics는 `isaac_sim/pad_motion.py`의 결정론적 코드가 담당한다.
Import한 URDF/USD는 appearance와 geometry provenance일 뿐 학습되거나 wheel physics를
사용하는 controller가 아니다. 설정된 carrier 치수는 0.720×0.500×0.345 m,
질량 75 kg, payload rating 120 kg, 속도 제한 1.0 m/s다.
