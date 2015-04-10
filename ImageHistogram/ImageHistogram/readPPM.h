
#define CREATOR "SHASHANK"
#define RGB_COMPONENT_COLOR 255


typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int width, height;
     PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *);
void writePPM(const char *, PPMImage *);
