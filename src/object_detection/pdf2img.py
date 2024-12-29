import pdf2image
import argparse
import logging
import os


def convert_pdf_to_image(output,pdf,format,dpi,prefix=""):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)

    # Convert the PDF to images
    images = pdf2image.convert_from_path(
        pdf,
        dpi=dpi,  # standard dpi used by pdfplumber is 72
        fmt=format)

    # Save the images
    for i, image in enumerate(images):
        if len(prefix)>0:
            image.save(
                os.path.join(
                    output, f"{prefix}_page_{i}.png"))
        else:
            image.save(
                os.path.join(
                    output, f"page_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf",
                        required=True,
                        help="Path to the PDF file")
    parser.add_argument("--output",
                        required=False,
                        default="images",
                        help="Path to the output folder")
    parser.add_argument("--format",
                        required=False,
                        default="png",
                        help="Output image format")
    parser.add_argument("--dpi",
                        required=False,
                        default=72,
                        help="Output image format")

    args = parser.parse_args()
    convert_pdf_to_image(args.output,args.pdf,args.format,args.dpi)

    # Create the output folder if it doesn't exist
    # if not os.path.exists(args.output):
    #     os.makedirs(args.output)

    # # Convert the PDF to images
    # images = pdf2image.convert_from_path(
    #     args.pdf,
    #     dpi=args.dpi,  # standard dpi used by pdfplumber is 72
    #     fmt=args.format)

    # # Save the images
    # for i, image in enumerate(images):
    #     image.save(
    #         os.path.join(
    #             args.output, f"page_{i}.png"))

    logging.info(f"PDF converted to images and saved at {args.output}")
