from ui import gradio_interface
import gradio as gr


def main():
    inputs = [
        gr.Image(type="numpy", label="Source Image"),
        gr.Image(type="numpy", label="Target Image"),
        gr.Slider(0.5, 2.0, value=1.0, label="Scale"),
        gr.Slider(-500, 500, value=0, label="Translation X"),
        gr.Slider(-500, 500, value=0, label="Translation Y"),
        gr.Slider(-180, 180, value=0, label="Rotation Angle"),
    ]

    outputs = gr.Image(type="numpy", label="Blended Result")

    gr.Interface(
        fn=gradio_interface,
        inputs=inputs,
        outputs=outputs,
        title="No Cameraman Left Behind",
    ).launch()


if __name__ == "__main__":
    main()
