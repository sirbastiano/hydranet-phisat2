from hydranet import available_student_presets, load_student, load_teacher


def main() -> None:
    print("Student presets:", available_student_presets())

    student = load_student()
    print("Student model:", student)

    teacher = load_teacher(
        task="classification",
        pretrained_path=None,
        input_dim=3,
        output_dim=10,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        img_size=224,
    )
    print("Teacher model:", teacher.__class__.__name__)


if __name__ == "__main__":
    main()
